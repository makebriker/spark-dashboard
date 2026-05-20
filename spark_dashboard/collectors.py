"""Metric collectors — wraps SparkView layers + disk info for the web dashboard."""

from __future__ import annotations

import dataclasses
import platform
import time
from datetime import datetime

import psutil

# SparkView layers (requires PYTHONPATH to include ~/sparkview)
from sparkview.layers import power_rails
from sparkview.layers.cpu import get_cpu_info
from sparkview.layers.gpu import get_gpu_info
from sparkview.layers.info import get_info
from sparkview.layers.memory import get_memory
from sparkview.layers.network import get_net_info as _sv_get_net_info
from sparkview.layers.power import get_power_info
from sparkview.layers.pressure import get_pressure
from sparkview.layers.throttle import get_throttle_info

SKIP_IFACES = {"lo", "docker0"}
SKIP_PREFIXES = ("br-",)
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
    """Fallback network collector for non-ConnectX-7 interfaces."""
    global _prev_net_time

    now = time.monotonic()
    elapsed = now - _prev_net_time if _prev_net_time > 0 else 1.0
    _prev_net_time = now

    results = []
    net_io = psutil.net_io_counters(pernic=True)
    from pathlib import Path as _Path

    net_sys = _Path("/sys/class/net")

    for iface_path in sorted(net_sys.iterdir()):
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


def _safe_call(func, *args, default=None, **kwargs):
    try:
        result = func(*args, **kwargs)
        return result
    except Exception:
        return default


def get_metrics() -> dict:
    gpus = _safe_call(get_gpu_info, default=[])
    mem = _safe_call(get_memory, default={})
    cpu = _safe_call(get_cpu_info, default={})
    psi = _safe_call(get_pressure, default={})
    throttle = _safe_call(get_throttle_info, gpus, default=[])
    power = _safe_call(get_power_info, default={"available": False})
    info = _safe_call(get_info, default={})

    # Try SparkView's ConnectX-7 aware collector first, then fallback to generic
    net = _safe_call(_sv_get_net_info, default=[])
    if not net:
        net = get_net_info_fallback()

    # power_rails.read() returns PowerRailsData dataclass or None
    rails_data = _safe_call(power_rails.read, default=None)
    rails = dataclasses.asdict(rails_data) if rails_data else None

    disk = _safe_call(get_disk_info, default=[])

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
    }