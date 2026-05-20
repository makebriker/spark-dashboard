"""Metric collectors — wraps SparkView layers + disk info for the web dashboard."""

from __future__ import annotations

import dataclasses
import platform
from datetime import datetime
from pathlib import Path

import psutil

# SparkView layers (requires PYTHONPATH to include ~/sparkview)
from sparkview.layers import power_rails
from sparkview.layers.cpu import get_cpu_info
from sparkview.layers.gpu import get_gpu_info
from sparkview.layers.info import get_info
from sparkview.layers.memory import get_memory
from sparkview.layers.network import get_net_info
from sparkview.layers.power import get_power_info
from sparkview.layers.pressure import get_pressure
from sparkview.layers.throttle import get_throttle_info


def _gi(b: int | None) -> str:
    if b is None:
        return "N/A"
    return f"{b / (1024**3):.1f}Gi"


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
    net = _safe_call(get_net_info, default=[])

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