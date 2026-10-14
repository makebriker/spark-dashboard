"""Supplemental network collection for the spark-dashboard web UI.

SparkView's own network layer (sparkview.layers.network) only discovers
ConnectX-7 interfaces — the four QSFP ports on the mlx5 driver. On GB10 and
GB10-2 that silently hides the onboard ethernet (enP7s7), which carries all LAN
traffic, and the WiFi adapter (wlP9s9).

This module wraps that layer: it keeps the CX7 entries exactly as SparkView
produces them, appends the remaining physical interfaces using the same schema,
and attaches IPv4 addressing to every entry.

It also substitutes **RDMA** throughput for the QSFP ports. NCCL runs RoCE v2
with kernel bypass, so tensor-parallel traffic never enters the kernel network
stack and cannot appear in the netdev counters psutil reads — those report the
TCP sideband only (~14 KB/s of gloo bootstrap and scheduler chatter) while the
link is really moving ~1.3 Gbit/s. Measured on a live TP=2 vLLM deployment the
understatement was ~12,000x. For interfaces that have an InfiniBand device,
``rx_rate``/``tx_rate`` therefore come from the IB port counters; the netdev
values are preserved under ``netdev_rx_rate``/``netdev_tx_rate`` rather than
discarded.

Like the layer it wraps: sysfs and psutil only, no subprocess, no ethtool.
"""

from __future__ import annotations

import pathlib
import socket
import time

import psutil

from sparkview.layers.network import get_net_info as _sv_get_net_info

NET_SYS = pathlib.Path("/sys/class/net")

# Also used by collectors.get_net_info_fallback()
SKIP_IFACES = {"lo", "docker0"}
SKIP_PREFIXES = ("br-", "veth", "virbr", "tun", "tap")

# IB port counters are expressed in 4-byte lanes, not bytes. Forgetting this
# multiplier yields a number 4x too small — plausible enough to go unnoticed,
# which is exactly why it is named here rather than inlined as a literal.
IB_LANE_BYTES = 4

_prev_stats: dict[str, dict] = {}
_prev_time: float = 0.0

_prev_rdma: dict[str, dict] = {}
_prev_rdma_time: float = 0.0


def is_physical(iface: str) -> bool:
    """True for interfaces backed by real hardware.

    Virtual interfaces (lo, docker0, bridges, veth pairs) have no `device`
    symlink in sysfs.
    """
    return (NET_SYS / iface / "device").exists()


def read_operstate(iface: str) -> str:
    try:
        return (NET_SYS / iface / "operstate").read_text().strip().upper()
    except OSError:
        return "UNKNOWN"


def read_speed(iface: str) -> int | None:
    """Negotiated link speed in Mbps, or None when the link is down."""
    try:
        val = int((NET_SYS / iface / "speed").read_text().strip())
        return val if val > 0 else None
    except (OSError, ValueError):
        return None


def read_address(iface: str) -> str:
    try:
        return (NET_SYS / iface / "address").read_text().strip()
    except OSError:
        return ""


def rdma_device(iface: str) -> str | None:
    """Name of the InfiniBand device behind an interface, or None.

    ``/sys/class/net/<iface>/device/infiniband/<hca>`` resolves only for
    RDMA-capable NICs, which makes its mere existence the discriminator: it
    yields ``rocep1s0f1`` for enp1s0f1np1 and ``roceP2p1s0f1`` for
    enP2p1s0f1np1, and is absent for the onboard ethernet and WiFi.
    """
    try:
        return next(iter(sorted(p.name for p in (NET_SYS / iface / "device" / "infiniband").iterdir())))
    except (OSError, StopIteration):
        return None


def read_rdma_counters(hca: str) -> tuple[int, int] | None:
    """(bytes transmitted, bytes received) for an HCA's first port.

    Returns bytes, having already applied the 4-byte lane multiplier.
    """
    try:
        ports = sorted((pathlib.Path("/sys/class/infiniband") / hca / "ports").iterdir())
    except OSError:
        return None
    for port in ports:
        try:
            counters = port / "counters"
            xmit = int((counters / "port_xmit_data").read_text().strip())
            rcv = int((counters / "port_rcv_data").read_text().strip())
        except (OSError, ValueError):
            continue
        return xmit * IB_LANE_BYTES, rcv * IB_LANE_BYTES
    return None


def apply_rdma_rates(entries: list[dict]) -> None:
    """Replace netdev throughput with RDMA throughput where one exists.

    Netdev counters cannot observe kernel-bypass traffic, so for RoCE ports
    they report the TCP sideband and nothing else. The original values are kept
    under ``netdev_*`` so the sideband stays inspectable.
    """
    global _prev_rdma_time  # noqa: PLW0603

    now = time.monotonic()
    elapsed = now - _prev_rdma_time if _prev_rdma_time > 0 else 1.0
    _prev_rdma_time = now

    for entry in entries:
        iface = entry["iface"]
        entry["rdma_hca"] = None
        entry["transport"] = "netdev"

        hca = rdma_device(iface)
        if hca is None:
            continue
        counters = read_rdma_counters(hca)
        if counters is None:
            continue

        tx_bytes, rx_bytes = counters
        prev = _prev_rdma.get(iface)
        _prev_rdma[iface] = {"tx": tx_bytes, "rx": rx_bytes}

        entry["rdma_hca"] = hca
        entry["transport"] = "rdma"
        entry["netdev_rx_rate"] = entry.get("rx_rate", 0.0)
        entry["netdev_tx_rate"] = entry.get("tx_rate", 0.0)

        if prev is None:
            entry["rx_rate"] = 0.0
            entry["tx_rate"] = 0.0
            continue

        # A negative delta means the counter was reset (driver reload, port
        # bounce). Report zero for that cycle rather than a spike.
        rx_delta = rx_bytes - prev["rx"]
        tx_delta = tx_bytes - prev["tx"]
        entry["rx_rate"] = rx_delta / elapsed if rx_delta >= 0 else 0.0
        entry["tx_rate"] = tx_delta / elapsed if tx_delta >= 0 else 0.0


def netmask_to_prefix(netmask: str | None) -> int | None:
    """Convert a dotted-quad netmask to a prefix length."""
    if not netmask:
        return None
    try:
        packed = socket.inet_aton(netmask)
    except OSError:
        return None
    return bin(int.from_bytes(packed, "big")).count("1")


def get_ipv4(addrs: list) -> dict | None:
    """First IPv4 address of an interface as {address, prefix}, or None."""
    for addr in addrs:
        if addr.family == socket.AF_INET:
            return {"address": addr.address, "prefix": netmask_to_prefix(addr.netmask)}
    return None


def get_extra_net_info(exclude: set[str]) -> list[dict]:
    """Physical interfaces SparkView's CX7 collector does not report.

    Emits the same schema as sparkview.layers.network.get_net_info() so the
    frontend needs no per-interface special-casing.
    """
    global _prev_time  # noqa: PLW0603

    now = time.monotonic()
    elapsed = now - _prev_time if _prev_time > 0 else 1.0
    _prev_time = now

    net_io = psutil.net_io_counters(pernic=True)

    results = []
    for iface_path in sorted(NET_SYS.iterdir()):
        iface = iface_path.name
        if iface in exclude or iface in SKIP_IFACES or iface.startswith(SKIP_PREFIXES):
            continue
        if not is_physical(iface) or iface not in net_io:
            continue

        counters = net_io[iface]
        prev = _prev_stats.get(iface, {})
        rx_rate = max(0, (counters.bytes_recv - prev.get("rx", counters.bytes_recv)) / elapsed) if prev else 0.0
        tx_rate = max(0, (counters.bytes_sent - prev.get("tx", counters.bytes_sent)) / elapsed) if prev else 0.0

        _prev_stats[iface] = {"rx": counters.bytes_recv, "tx": counters.bytes_sent}

        results.append(
            {
                "iface": iface,
                "state": read_operstate(iface),
                "speed_mbps": read_speed(iface),
                "rx_rate": rx_rate,
                "tx_rate": tx_rate,
                "rx_errors": counters.errin,
                "tx_errors": counters.errout,
                "rx_dropped": counters.dropin,
                "address": read_address(iface),
                "primary": False,
            }
        )

    return results


def get_net_info_all() -> list[dict]:
    """CX7 fabric ports plus onboard ethernet and WiFi, each with IPv4.

    QSFP ports report RDMA throughput; everything else reports netdev.

    Must be called once per metrics cycle: this module, its RDMA sampler and
    the SparkView layer all derive throughput from deltas between consecutive
    calls.
    """
    cx7 = _sv_get_net_info()
    seen = {entry["iface"] for entry in cx7}

    results = cx7 + get_extra_net_info(exclude=seen)

    # Applied to every entry, not just the appended ones: the RoCE ports are
    # exactly the interfaces SparkView's own layer produces.
    apply_rdma_rates(results)

    if_addrs = psutil.net_if_addrs()
    for entry in results:
        entry["ipv4"] = get_ipv4(if_addrs.get(entry["iface"], []))

    return results
