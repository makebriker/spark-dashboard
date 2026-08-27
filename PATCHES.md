# Patches over upstream SparkView

This dashboard reads its metrics from the [SparkView](https://github.com/parallelArchitect/sparkview)
collection layers, installed separately at `~/sparkview/`. Three of those layers
produce wrong or incomplete numbers on GB10 hardware. Rather than patch the upstream
clone — which is a real git clone and would conflict on the next `git pull` — the
corrections live here as wrapper modules that call upstream's own primitives and fix
only the specific final step that is wrong.

Two modules do this: `spark_dashboard/net_extra.py` and
`spark_dashboard/gpu_throttle_fix.py`. `collectors.py` imports from them instead of
importing the upstream functions directly.

---

## 1. Network discovery misses everything that isn't a QSFP port

**`net_extra.py` → `get_net_info_all()`**

`sparkview/layers/network.py` only discovers ConnectX-7 interfaces — those listed in
`CX7_PRIMARY` or bound to the `mlx5` driver. `collectors.py` falls back to a generic
collector *only when that list comes back empty*, which never happens: on a GX10 all
four QSFP ports always exist in sysfs.

The result is that the onboard ethernet (`enP7s7`) — which carries all LAN traffic,
including SSH — and the WiFi adapter (`wlP9s9`) were silently absent from the NETWORK
card, and no interface showed an IP address at all.

`get_net_info_all()` calls the upstream layer once for the CX7 ports (preserving its
throughput-delta state), appends every remaining *physical* interface in the same
schema, and attaches `ipv4: {address, prefix} | None` to every entry. Pure sysfs plus
`psutil` — no subprocess, no `ethtool`/`iw`/`nmcli` — matching the design constraint
of the layer it wraps.

Expect six interfaces on a GX10: four QSFP, the onboard ethernet, and WiFi.

## 2. QSFP throughput reads ~0 while RDMA traffic saturates the link

**`net_extra.py` → `apply_rdma_rates()`**

The NETWORK card reported the 200G QSFP ports at a few KB/s while they were actively
carrying a distributed-inference workload.

The counters were not broken — they were structurally blind. RoCE v2 uses **kernel
bypass**: the NIC reads and writes application memory directly, and nothing it sends
ever enters the kernel network stack that `psutil.net_io_counters()` reads. Those
counters see only the TCP sideband — bootstrap and control chatter — so they
understate real throughput by orders of magnitude.

`get_net_info_all()` now runs `apply_rdma_rates()` over **every** entry, not just the
appended ones, since the RoCE ports are exactly the interfaces the upstream layer
produces. Per interface:

- `rdma_device()` resolves `/sys/class/net/<iface>/device/infiniband/<hca>`. Its mere
  existence is the discriminator — present for the QSFP ports, absent for onboard
  ethernet and WiFi.
- If present, `rx_rate`/`tx_rate` are recomputed from
  `ports/<n>/counters/port_{xmit,rcv}_data`. The netdev values are preserved under
  `netdev_rx_rate`/`netdev_tx_rate` rather than discarded, and the entry gains
  `transport: "rdma"|"netdev"` and `rdma_hca`.
- If absent, the entry is untouched and keeps its netdev rates.

**Those counters are in 4-byte lanes, not bytes** — hence `IB_LANE_BYTES = 4`, named
rather than inlined, because a 4x error here still looks plausible. A negative delta
(driver reload, port bounce) reports 0 for that cycle instead of a spike.

Validated against a manual `port_xmit_data` delta over a 10 s window: 0.0% error.

## 3. GPU memory reports system memory on coherent-UMA parts

**`gpu_throttle_fix.py` → `get_gpu_info_fixed()`**

`sparkview/layers/gpu.py` reports `mem_used = vm.total - vm.available` — system-wide
committed memory — labeled as GPU memory. On GB10 the symptom is unmistakable: the
GPU card and the MEMORY card show *identical* figures, and the GPU appears pinned near
100% memory at idle.

`nvidia-smi --query-gpu=memory.used` genuinely is `Not Supported` on GB10, which is
presumably why the fallback exists. But
`nvidia-smi --query-compute-apps=pid,used_memory` — the per-process NVML compute-app
query that nvitop's `Device.processes()` already calls — works and returns real
numbers. Upstream was already fetching this data for the process list, just not using
it for the device-level total.

## 4. The throttle-reason bit table doesn't match NVML

**`gpu_throttle_fix.py` → `get_throttle_info_fixed()`**

`sparkview/layers/throttle.py`'s `PROBLEM_THROTTLE_BITS` does not match NVML's actual
`nvmlClocksEventReason*` bit assignments, checked against `pynvml` directly on GB10:

| Bit | NVML actual | `throttle.py` said |
|---|---|---|
| `0x04` | `SwPowerCap` | *missing* |
| `0x08` | `HwSlowdown` | `HW_SLOWDOWN` ✓ |
| `0x10` | `SyncBoost` | `SW_THERMAL_SLOWDOWN` ✗ |
| `0x20` | `SwThermalSlowdown` | `HW_THERMAL_SLOWDOWN` ✗ |
| `0x40` | `HwThermalSlowdown` | *missing* |
| `0x80` | `HwPowerBrakeSlowdown` | ✓ |
| `0x100` | `DisplayClockSetting` | `SW_POWER_CAP` ✗ |

This matters: a reported `hw_thermal_slowdown` was bit `0x20` — a genuine *software*
thermal slowdown — mislabeled as a hardware fault. Over ~3,654 s of measured GPU busy
time, real hardware thermal slowdown accounted for 2.6 s (0.07%). There was no
hardware thermal problem at all.

The corrected table drops `0x10` and `0x100` (SyncBoost and DisplayClockSetting —
neither indicates a real performance problem on a single-GPU box) and adds the two
that were silently missing.

Both fixes re-implement their function using upstream's own lower-level primitives
(`is_coherent_uma`, `query_nvidia_smi`, `NVITOP_AVAILABLE`/`Device`), so they track
upstream's internals rather than duplicating them — only the two buggy final steps are
corrected.

---

## Removed: the POWER RAILS card

`power_rails.read()` returns `None` on GB10 — the `spark_hwmon` telemetry it depends
on is not available — so the card's rich view never rendered. Its fallback branch
displayed a single "Power Draw" row taken from NVML, which is the same number the GPU
card already shows. The card was pure duplication and has been removed.

`/api/metrics` still emits the `power_rails` key, so anything scraping the API is
unaffected. If the rail telemetry becomes available on your hardware, the collector
side is intact and only the frontend card needs restoring.

**Upgrade note.** Removing a card changes `index.html` and `dashboard.js` together,
and the two must stay in lockstep — a cached `dashboard.js` looking up an id the new
HTML no longer has will throw. Originally none of the three frontend files sent a
`Cache-Control` header, so browsers applied heuristic freshness to `etag`/
`last-modified` and could skip revalidation; the symptom was NETWORK and DISK
rendering empty while every card above them was fine, because the throw aborted
`updateAll()` partway down the list. They are now served `Cache-Control: no-cache`,
and `updateAll()` renders each card isolated so one failure cannot blank the rest.

## Verifying the patches

```bash
curl -s localhost:8500/api/metrics | python3 -c "
import json,sys
d = json.load(sys.stdin)
g = round(d['gpu'][0]['mem_used'] / 2**20)
m = round(d['memory']['used'] / 2**20)
print('gpu mem  (MiB):', g)
print('sys mem  (MiB):', m, '<- must differ from the line above')
print('throttle:', d['throttle'][0]['throttle_reasons'])
for e in sorted(d['network'], key=lambda x: x['iface']):
    print(f\"{e['iface']:16}{e.get('transport','?'):8}{e['tx_rate']*8/1e9:8.3f} Gbit/s  {e.get('ipv4')}\")
"
```

Expect: `gpu mem` clearly **below** `sys mem` (equality means patch 3 is not active),
six interfaces, an `ipv4` on the onboard ethernet, and `transport: rdma` on the QSFP
ports.
