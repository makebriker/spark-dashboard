# SparkView Dashboard

Web dashboard for the NVIDIA Grace Blackwell GB10 (ASUS GX10) that exposes SparkView metrics in a browser with an NVIDIA-inspired dark UI.

![Dashboard](https://img.shields.io/badge/port-8500-76B900?style=flat-square)

## Features

- **GPU** — Utilization, temperature, power draw, UMA-aware memory, top processes
- **Memory** — RAM and swap usage with GB10 unified memory awareness
- **CPU** — Per-core utilization, temperature, frequency
- **Clock/Throttle** — Clock speed, P-state, throttle status (IDLE/PASS/LOCKED/THROTTLED)
- **Power Rails** — GPU draw, peak, DC input, SysPL1 cap, PROCHOT, PL level, Tj rise
- **PSI Pressure** — Memory pressure levels (LOW/MOD/HIGH/CRITICAL)
- **Network** — ConnectX-7 interface state, TX/RX rates, errors
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

```bash
./deploy.sh
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
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

## License

MIT