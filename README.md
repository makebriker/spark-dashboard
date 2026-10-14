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
- **Inference (vLLM)** — detected automatically when a vLLM server runs on the node:
  decode throughput (aggregate and per stream), prefill rate, running/waiting requests,
  KV-cache use, TTFT, inter-token latency, prefix-cache hit rate and more, with a
  decode-throughput sparkline
- **Multi-node** — up to 4 Sparks on one board, side by side or stacked
- **Configurable** — which cards appear, in what order, per-card filters, which vLLM
  numbers to show — from one TOML file, applied live without a restart

Auto-refreshes every 2 seconds by default. No external dependencies — works entirely offline on the LAN.

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

## Board configuration

Copy the example and edit it — every key is optional and documented inline:

```bash
cp config.example.toml config.toml
```

The file is looked up at `$SPARK_DASHBOARD_CONFIG`, then `./config.toml` (the deployed
checkout), then `~/.config/spark-dashboard/config.toml`. It is gitignored, so
`deploy.sh` never overwrites it, and it is re-read whenever it changes. A file that
fails to parse is logged and the previous config stays in effect. With no file, the
board looks like v1 plus the vLLM card when a server is found.

### Several Sparks on one board

Every Spark runs this dashboard as usual. Pick one to be the board and list the others
as nodes in *its* config:

```toml
[[nodes]]
name = "gb10"
url = "local"                      # this machine

[[nodes]]
name = "gb10-2"
url = "http://192.0.2.11:8500"     # another Spark's dashboard
```

The board fetches each node's `/api/metrics` server-side, so the browser only needs to
reach the board. An unreachable node is marked red and its last values greyed out;
the other nodes are unaffected. `layout = "columns"` puts nodes side by side (suits up
to 4 on a wide screen); `"rows"` gives each node a full-width band.

### vLLM

With `[vllm] enabled = "auto"` the node scans its process table for `vllm serve` /
`vllm.entrypoints.openai.api_server`, takes `--port` from the command line (default
8000), and scrapes `http://127.0.0.1:<port>/metrics`. Docker containers are found as
long as the port is the same on the host; otherwise list the URL under `endpoints`.
For tensor-parallel deployments over Ray only the head node runs the API server, so
only its column shows the card. Rates and latency means are taken over a sliding
`rate_window_s` window; idle values show as `—`.

## Deploy configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | _(required)_ | IP or hostname of the GB10 device |
| `USER_REMOTE` | `lesha` | SSH user on the GB10 device |
| `BRANCH` | `main` | Branch to deploy |
| `PYTHONPATH` | `~/sparkview` | Path to SparkView package |
| Port | `8500` | Configured in systemd unit |

## API

- `GET /api/metrics` — this node's metrics, including a `vllm` block. Sampled at most
  once a second and shared between callers. This is what other boards fetch.
- `GET /api/board` — what the page renders: the frontend-relevant config plus every
  configured node's metrics (`{name, url, ok, error?, data?}`), in one request.

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
                      ├── spark_dashboard.config (config.toml, hot-reloaded)
                      ├── spark_dashboard.collectors (metric aggregation)
                      ├── spark_dashboard.vllm_probe (vLLM detection + /metrics)
                      ├── other nodes' /api/metrics (multi-node boards)
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