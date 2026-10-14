"""SparkView Web Dashboard — FastAPI application."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from .config import MAX_NODES, get_config

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SparkView Dashboard", docs_url=None, redoc_url=None)

# The frontend is three files that must stay in lockstep: a stale cached
# dashboard.js against a fresh index.html breaks at the first id that changed.
# Without an explicit Cache-Control, browsers apply heuristic freshness to
# etag/last-modified and can skip revalidation entirely. "no-cache" means
# revalidate every time -- these are a few KB and answer 304.
NO_CACHE = {"Cache-Control": "no-cache"}
NO_STORE = {"Cache-Control": "no-store"}

# Nothing on a request path collects or fetches anything. Two background loops
# do all the work on a fixed cadence and every HTTP handler only reads their
# latest snapshot, so load is the same for one viewer or fifty:
#
#   local sampler  this host's hardware, every board.refresh_ms (only on a
#                  host that has the collectors -- a GB10, not the Mac board)
#   board poller   every configured node, every board.refresh_ms
#
# vLLM has its own loop in vllm_probe (scrape_interval_s), for the same reason.

REMOTE_TIMEOUT_S = 3.0
# A node that misses a poll keeps its last data, flagged stale, this long.
STALE_MAX_S = 30.0


class _Snapshot:
    def __init__(self):
        self.value = None
        self.ready = threading.Event()

    def set(self, value):
        self.value = value
        self.ready.set()

    async def get(self, wait_s: float = 5.0):
        if not self.ready.is_set():
            await asyncio.to_thread(self.ready.wait, wait_s)
        return self.value


_local = _Snapshot()     # {"ok": bool, "data"| "error", "at": monotonic}
_board = _Snapshot()     # [node payload, ...]
_started = False


def _every(period_s, fn, name):
    """Run fn on a fixed cadence (start-to-start) in a daemon thread."""
    def loop():
        while True:
            t0 = time.monotonic()
            try:
                fn()
            except Exception as err:  # never let a loop die
                print(f"{name}: {type(err).__name__}: {err}", flush=True)
            time.sleep(max(0.2, period_s() - (time.monotonic() - t0)))
    threading.Thread(target=loop, name=name, daemon=True).start()


def _refresh_s() -> float:
    return get_config()["board"]["refresh_ms"] / 1000


def _start_local_sampler() -> bool:
    try:
        # The collectors need SparkView and NVIDIA hardware; a board whose
        # nodes are all remote (the Mac) has neither and samples nothing.
        from .collectors import get_metrics
    except ImportError as err:
        _local.set({"ok": False, "error": f"no local collectors on this host ({err})"})
        return False

    from .vllm_probe import start_sampler
    start_sampler(lambda: get_config()["vllm"])

    def sample():
        _local.set({"ok": True, "data": get_metrics(), "at": time.monotonic()})

    _every(_refresh_s, sample, "local-sampler")
    return True


def _fetch_remote(url: str) -> dict:
    with urllib.request.urlopen(url + "/api/metrics", timeout=REMOTE_TIMEOUT_S) as resp:
        return json.loads(resp.read())


_last_good: dict[str, tuple[float, dict]] = {}


def _poll_node(node: dict) -> dict:
    out = {"name": node["name"], "url": node["url"], "cards": node.get("cards")}
    try:
        if node["url"] == "local":
            snap = _local.value or {"ok": False, "error": "local sampler not ready"}
            if not snap["ok"]:
                raise RuntimeError(snap["error"])
            data = snap["data"]
        else:
            data = _fetch_remote(node["url"])
        _last_good[node["url"]] = (time.monotonic(), data)
        out.update(ok=True, data=data)
    except Exception as err:
        error = f"{type(err).__name__}: {err}"
        last = _last_good.get(node["url"])
        if last and time.monotonic() - last[0] < STALE_MAX_S:
            out.update(ok=True, data=last[1], stale=True, error=error,
                       age_s=round(time.monotonic() - last[0]))
        else:
            out.update(ok=False, error=error)
    return out


def _start_board_poller():
    pool = ThreadPoolExecutor(max_workers=MAX_NODES, thread_name_prefix="node-fetch")

    def poll():
        nodes = get_config()["nodes"]
        _board.set(list(pool.map(_poll_node, nodes)))

    _every(_refresh_s, poll, "board-poller")


@asynccontextmanager
async def lifespan(_app):
    global _started
    if not _started:  # idempotent under reload/test clients
        _started = True
        _start_local_sampler()
        _start_board_poller()
    yield


app.router.lifespan_context = lifespan

def frontend_config(cfg: dict) -> dict:
    return {
        "board": cfg["board"],
        "cards": cfg["cards"],
        "vllm": {k: cfg["vllm"][k] for k in ("enabled", "metrics", "sparkline", "placement")},
    }


def _git_version() -> str:
    """Return the short git hash of the app repo, or 'dev' if not a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=STATIC.parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "dev"


JS_VERSION = _git_version()


@app.get("/api/metrics")
async def metrics():
    """This node's latest sample. What other boards and metrics-poller read."""
    snap = await _local.get()
    if not snap or not snap["ok"]:
        error = snap["error"] if snap else "local sampler not ready"
        return JSONResponse(content={"error": error}, status_code=503, headers=NO_STORE)
    return JSONResponse(content=snap["data"], headers=NO_STORE)


@app.get("/api/board")
async def board():
    """Everything the page renders, in one request: config plus every node."""
    cfg = get_config()
    nodes = await _board.get() or []
    return JSONResponse(
        content={"config": frontend_config(cfg), "nodes": nodes}, headers=NO_STORE
    )


@app.get("/style.css")
async def style():
    return FileResponse(
        STATIC / "style.css", media_type="text/css", headers=NO_CACHE
    )


@app.get("/dashboard.js")
async def javascript():
    return FileResponse(
        STATIC / "dashboard.js",
        media_type="application/javascript",
        headers=NO_CACHE,
    )


@app.get("/")
async def index():
    html = (STATIC / "index.html").read_text()
    html = html.replace("__JS_VERSION__", JS_VERSION)
    return HTMLResponse(content=html, headers=NO_CACHE)
