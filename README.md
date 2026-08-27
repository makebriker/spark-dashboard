# SparkView Dashboard

Web dashboard for the NVIDIA Grace Blackwell GB10 (ASUS GX10) that exposes SparkView metrics in a browser with an NVIDIA-inspired dark UI.

![Dashboard](https://img.shields.io/badge/port-8500-76B900?style=flat-square)

## Features

- **GPU** — Utilization, temperature, power draw, true UMA-aware memory, top processes
- **Memory** — RAM and swap usage, plus PSI memory-pressure levels (LOW/MOD/HIGH/CRITICAL)
- **CPU** — Per-core utilization, temperature, frequency
- **Clock/Throttle** — Clock speed, P-state, throttle status (IDLE/PASS/LOCKED/THROTTLED)
- **Network** — Every physical interface with its IPv4 address; **RDMA-aware** TX/RX
  rates on the ConnectX-7 QSFP ports, which kernel counters cannot see
- **Disk** — Per-mountpoint usage with color thresholds

Auto-refreshes every 2 seconds. No external dependencies — works entirely offline on the LAN.

## Prerequisites

- Python 3.12+
- [SparkView](https://github.com/parallelArchitect/sparkview) installed at `~/sparkview/` (for the metric collection layers)
- `spark_hwmon` kernel module (optional, for power rail telemetry)

## Quick Start

### Install & Run Manually

```bash
cd ~/spark-dashboard
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=~/sparkview .venv/bin/uvicorn spark_dashboard.app:app --host 0.0.0.0 --port 8500
```

Open http://<your-ip>:8500

### Deploy with systemd (auto-start)

The deployed directory is a git checkout of this repository, so deploying is a fetch
plus a restart. `deploy.sh` also converts a non-git directory in place on first run,
taking a backup first and preserving the gitignored `.venv/`:

```bash
HOST=<ip-or-hostname> ./deploy.sh
```

It installs the systemd user unit, restarts it, verifies `/api/metrics` answers, and
prints `git status` so any local drift on the box is visible.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | _(required)_ | IP or hostname of the GB10 device |
| `USER_REMOTE` | `lesha` | SSH user on the GB10 device |
| `BRANCH` | `main` | Branch to deploy |
| `PYTHONPATH` | `~/sparkview` | Path to SparkView package |
| Port | `8500` | Configured in systemd unit |

## API

`GET /api/metrics` — Returns JSON with all metrics sampled at request time.

## Stopping

```bash
systemctl --user stop spark-dashboard      # temporary
systemctl --user disable spark-dashboard   # permanent
```

## Architecture

```
Browser ──HTTP──> uvicorn (FastAPI)
                      │
                      ├── spark_dashboard.app (routes)
                      ├── spark_dashboard.collectors (metric aggregation)
                      │
                      ├── spark_dashboard.net_extra ─────┐  local fixes over
                      ├── spark_dashboard.gpu_throttle_fix ┤  upstream — see
                      │                                  ─┘  PATCHES.md
                      └── sparkview.layers.* (SparkView's metric modules)
                              ├── gpu.py (nvitop/NVML)
                              ├── memory.py (psutil)
                              ├── cpu.py (psutil)
                              ├── throttle.py (nvidia-smi)
                              ├── power_rails.py (sysfs hwmon)
                              ├── pressure.py (/proc/pressure)
                              ├── network.py (sysfs)
                              └── info.py (nvitop + /proc)
```

`net_extra.py` and `gpu_throttle_fix.py` wrap three upstream SparkView layers that
return wrong or incomplete numbers on GB10 — missing network interfaces, RDMA traffic
invisible to kernel counters, GPU memory reported as system memory, and a throttle-bit
table that mislabels software thermal events as hardware faults. The upstream clone is
deliberately left untouched. **See [PATCHES.md](PATCHES.md).**

## License

MIT