"""Metric collectors — wraps SparkView layers + disk info for the web dashboard."""

from __future__ import annotations

import dataclasses
import logging
import platform
import time
from collections import deque
from datetime import datetime

import psutil

# SparkView layers (requires PYTHONPATH to include ~/sparkview)
from sparkview.layers import power_rails
from sparkview.layers.cpu import get_cpu_info
from sparkview.layers.info import get_info
from sparkview.layers.memory import get_memory
from sparkview.layers.power import get_power_info
from sparkview.layers.pressure import get_pressure

from spark_dashboard.config import get_config
from spark_dashboard.vllm_probe import get_vllm_info

# Wraps sparkview.layers.network — adds onboard ethernet, WiFi and IPv4
from spark_dashboard.net_extra import NET_SYS, SKIP_IFACES, SKIP_PREFIXES, get_net_info_all

# Wraps sparkview.layers.gpu / throttle — corrects mem_used and the
# throttle-bit table without touching the upstream clone (see PATCHES.md)
from spark_dashboard.gpu_throttle_fix import get_gpu_info_fixed as get_gpu_info
from spark_dashboard.gpu_throttle_fix import get_throttle_info_fixed as get_throttle_info

log = logging.getLogger("spark_dashboard.collectors")

_prev_net: dict[str, dict] = {}
_prev_net_time: float = 0.0


def get_disk_info() -> list[dict]:
    partitions = []
    for part in psutil.disk_partitions():
        if part.device.startswith("/dev/loop"):
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
            partitions.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            )
        except (PermissionError, OSError):
            continue
    return partitions


def get_net_info_fallback() -> list[dict]:
    """Generic netdev collector, used only if net_extra fails outright."""
    global _prev_net_time

    now = time.monotonic()
    elapsed = now - _prev_net_time if _prev_net_time > 0 else 1.0
    _prev_net_time = now

    results = []
    net_io = psutil.net_io_counters(pernic=True)

    for iface_path in sorted(NET_SYS.iterdir()):
        iface = iface_path.name
        if iface in SKIP_IFACES or iface.startswith(SKIP_PREFIXES):
            continue
        if iface not in net_io:
            continue

        counters = net_io[iface]
        operstate = "UNKNOWN"
        try:
            operstate = (iface_path / "operstate").read_text().strip().upper()
        except OSError:
            pass

        speed_mbps = None
        try:
            s = int((iface_path / "speed").read_text().strip())
            speed_mbps = s if s > 0 else None
        except OSError:
            pass

        prev = _prev_net.get(iface, {})
        rx_rate = max(0, (counters.bytes_recv - prev.get("rx", counters.bytes_recv)) / elapsed) if prev else 0.0
        tx_rate = max(0, (counters.bytes_sent - prev.get("tx", counters.bytes_sent)) / elapsed) if prev else 0.0

        _prev_net[iface] = {"rx": counters.bytes_recv, "tx": counters.bytes_sent}

        results.append({
            "iface": iface,
            "state": operstate,
            "speed_mbps": speed_mbps,
            "rx_rate": rx_rate,
            "tx_rate": tx_rate,
            "rx_errors": counters.errin,
            "tx_errors": counters.errout,
            "rx_dropped": counters.dropin,
            "primary": False,
        })

    return results


# rx_rate/tx_rate are averaged over this trailing window; the instantaneous
# value (since the previous sample, ~board.refresh_ms) moves to *_rate_now.
# metrics-poller stores one row every 15 s and dbt/Superset aggregate
# rx_rate/tx_rate, so a 15 s window gives each stored row continuous coverage
# instead of a 2 s glimpse -- regardless of how often this node is sampled.
NET_AVG_WINDOW_S = 15.0
_net_hist: dict[str, deque] = {}   # iface -> deque[(t, dt, rx_rate, tx_rate)]
_net_last_t: float = 0.0


def average_net_rates(entries: list[dict]) -> None:
    """Replace rx_rate/tx_rate with their time-weighted mean over the window.

    Each sample's rate covers exactly the interval since the previous sample,
    so sum(rate * dt) / sum(dt) is the true byte rate over the window -- the
    same number the raw counters would give, without needing them here.
    """
    global _net_last_t
    now = time.monotonic()
    # The first sample's rate has no real interval behind it; don't keep it.
    dt = now - _net_last_t if _net_last_t else 0.0
    _net_last_t = now
    seen = set()
    for e in entries:
        iface = e.get("iface")
        if iface is None:
            continue
        seen.add(iface)
        rx_now, tx_now = e.get("rx_rate") or 0.0, e.get("tx_rate") or 0.0
        e["rx_rate_now"], e["tx_rate_now"] = rx_now, tx_now
        hist = _net_hist.setdefault(iface, deque())
        if dt > 0:
            hist.append((now, dt, rx_now, tx_now))
        # Drop the oldest sample only while the rest still spans the whole
        # window, so coverage is never shorter than NET_AVG_WINDOW_S (a 15 s
        # poller row must not fall between two windows and miss a burst).
        span = sum(h[1] for h in hist)
        while hist and span - hist[0][1] >= NET_AVG_WINDOW_S:
            span -= hist.popleft()[1]
        if span > 0:
            # Count only the part of the oldest sample that falls inside the
            # window (its rate is uniform over its interval), so the window
            # is exactly NET_AVG_WINDOW_S once warm. Consecutive 15 s poller
            # rows then tile time instead of overlapping, and a burst in the
            # overlap is not counted twice (measured +36% before this).
            excess = max(0.0, span - NET_AVG_WINDOW_S)
            weights = [h[1] for h in hist]
            weights[0] -= excess
            span -= excess
            e["rx_rate"] = sum(w * h[2] for w, h in zip(weights, hist)) / span
            e["tx_rate"] = sum(w * h[3] for w, h in zip(weights, hist)) / span
        e["rate_window_s"] = round(span, 1)
    for gone in _net_hist.keys() - seen:
        del _net_hist[gone]


_reported: set[str] = set()


def _safe_call(func, *args, default=None, **kwargs):
    """Call a collector; on failure return default and log the error once.

    A silent failure just hides a card, which looks like a config choice.
    Logging once per distinct message keeps a permanently broken collector
    from flooding the journal every refresh.
    """
    try:
        return func(*args, **kwargs)
    except Exception as err:
        key = f"{getattr(func, '__name__', func)}: {type(err).__name__}: {err}"
        if key not in _reported:
            _reported.add(key)
            log.warning("collector failed (reported once): %s", key, exc_info=True)
        return default


def get_metrics() -> dict:
    gpus = _safe_call(get_gpu_info, default=[])
    mem = _safe_call(get_memory, default={})
    cpu = _safe_call(get_cpu_info, default={})
    psi = _safe_call(get_pressure, default={})
    throttle = _safe_call(get_throttle_info, gpus, default=[])
    power = _safe_call(get_power_info, default={"available": False})
    info = _safe_call(get_info, default={})

    # CX7 ports + onboard ethernet + WiFi, with IPv4; generic fallback if that fails
    net = _safe_call(get_net_info_all, default=[])
    if not net:
        net = get_net_info_fallback()
    _safe_call(average_net_rates, net)

    # power_rails.read() returns PowerRailsData dataclass or None
    rails_data = _safe_call(power_rails.read, default=None)
    rails = dataclasses.asdict(rails_data) if rails_data else None

    disk = _safe_call(get_disk_info, default=[])
    vllm = _safe_call(get_vllm_info, get_config()["vllm"], default={"detected": False, "servers": []})

    # Convert cpu_freq namedtuple to dict if present
    cpu_freq = cpu.get("freq")
    if cpu_freq is not None and hasattr(cpu_freq, "_asdict"):
        cpu["freq"] = dict(cpu_freq._asdict())

    # Convert per_core from tuple to list for JSON
    if "per_core" in cpu and isinstance(cpu["per_core"], tuple):
        cpu["per_core"] = list(cpu["per_core"])

    # Convert process gpu_mem from None
    for gpu in gpus:
        if "processes" in gpu:
            for proc in gpu["processes"]:
                if proc.get("gpu_mem") is None:
                    proc["gpu_mem"] = 0

    return {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "gpu": gpus,
        "memory": mem,
        "cpu": cpu,
        "pressure": psi,
        "throttle": throttle,
        "power": power,
        "power_rails": rails,
        "network": net,
        "info": info,
        "disk": disk,
        "vllm": vllm,
    }
