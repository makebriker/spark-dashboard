"""Wraps sparkview.layers.gpu / sparkview.layers.throttle — corrects two
independently-verified bugs without touching the upstream sparkview clone
(a real git clone left untouched per PATCHES.md, so a future `git pull`
does not clobber local fixes).

1. gpu.get_gpu_info(): on coherent-UMA devices (GB10), mem_used/mem_total
   substitute system-wide committed memory (vm.total - vm.available) for
   GPU memory — the GPU card ends up showing the exact same number as the
   MEMORY card. Sum of real per-process GPU allocations is used instead
   (NVML's per-process compute-app query, which nvitop's
   Device.processes()[pid].gpu_memory() already exposes and which *is*
   supported on GB10, unlike the device-level memory counter).

2. throttle.get_throttle_info(): PROBLEM_THROTTLE_BITS maps NVML
   clocks-event-reason bits to the wrong names (verified against pynvml
   on-device 2026-07-31): 0x10 SyncBoost was reported as
   SW_THERMAL_SLOWDOWN, 0x20 SwThermalSlowdown as HW_THERMAL_SLOWDOWN,
   0x100 DisplayClockSetting as SW_POWER_CAP, and the real SwPowerCap
   (0x04) and HwThermalSlowdown (0x40) were never decoded at all. SyncBoost
   and DisplayClockSetting are dropped from the corrected table — neither
   indicates a real performance problem on a single-GPU box.
"""

from __future__ import annotations

import psutil

from sparkview.layers.gpu import NVITOP_AVAILABLE, is_coherent_uma
from sparkview.layers.throttle import CLOCK_THRESHOLD_MHZ, LOAD_GATE_PCT, query_nvidia_smi

if NVITOP_AVAILABLE:
    from nvitop import Device

PROBLEM_THROTTLE_BITS: dict[int, str] = {
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
}


def decode_throttle(bitmask: int) -> list[str]:
    return [name for bit, name in PROBLEM_THROTTLE_BITS.items() if bitmask & bit]


def has_problem_throttle(bitmask: int) -> bool:
    return any(bitmask & bit for bit in PROBLEM_THROTTLE_BITS)


def get_throttle_info_fixed(gpu_procs: list[dict] | None = None) -> list[dict]:
    """Same shape as sparkview.layers.throttle.get_throttle_info, with the
    corrected bit table above."""
    gpus = query_nvidia_smi()
    if not gpus:
        return []
    results = []
    for g in gpus:
        clk = g["clk_mhz"]
        clk_max = g["clk_max_mhz"]
        util = g["util_pct"] or 0
        raw = g["throttle_raw"]
        reasons = decode_throttle(raw)
        problem = has_problem_throttle(raw)
        pstate = g.get("pstate", "P0")
        if clk is None or clk_max is None:
            results.append({"index": g["index"], "available": False})
            continue
        locked = clk_max < 2900 and clk_max > 0 and abs(clk - clk_max) < 50
        if util < LOAD_GATE_PCT or pstate not in ("P0", "P1", "P2"):
            status = "IDLE"
        elif locked:
            status = "LOCKED"
        elif clk < CLOCK_THRESHOLD_MHZ or problem:
            status = "THROTTLED"
        else:
            status = "PASS"
        results.append(
            {
                "index": g["index"],
                "available": True,
                "status": status,
                "clk_mhz": clk,
                "clk_max_mhz": clk_max,
                "pstate": pstate,
                "power_w": g["power_w"],
                "util_pct": util,
                "throttle_reasons": reasons,
                "problem_throttle": problem,
                "locked": locked,
            }
        )
    return results


def get_gpu_info_fixed() -> list[dict]:
    """Same shape as sparkview.layers.gpu.get_gpu_info, except mem_used/
    mem_total on coherent-UMA devices report real GPU-resident allocation
    (sum of per-process NVML compute-app memory) instead of system-wide
    committed memory — that figure is already reported separately by the
    MEMORY card."""
    if not NVITOP_AVAILABLE:
        return []
    results = []
    for device in Device.all():
        try:
            uma = is_coherent_uma(device)
            vm = psutil.virtual_memory()
            mem = device.memory_info()
            procs = []
            for p in device.processes().values():
                try:
                    cpu = p.cpu_percent()
                    procs.append(
                        {
                            "pid": p.pid,
                            "user": p.username(),
                            "gpu_mem": p.gpu_memory(),
                            "cpu_pct": cpu,
                            "cmd": p.name(),
                        }
                    )
                except Exception:  # noqa: BLE001
                    continue
            if uma:
                mem_used = sum(p["gpu_mem"] or 0 for p in procs)
                mem_total = vm.total
            else:
                mem_used = mem.used
                mem_total = mem.total
            results.append(
                {
                    "index": device.index,
                    "name": device.name(),
                    "utilization": device.gpu_utilization(),
                    "temperature": device.temperature(),
                    "power": device.power_usage(),
                    "is_uma": uma,
                    "mem_total": mem_total,
                    "mem_used": mem_used,
                    "processes": sorted(procs, key=lambda x: x["gpu_mem"] or 0, reverse=True),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return results
