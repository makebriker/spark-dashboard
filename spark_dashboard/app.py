"""SparkView Web Dashboard — FastAPI application."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from .config import get_config

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SparkView Dashboard", docs_url=None, redoc_url=None)

# The frontend is three files that must stay in lockstep: a stale cached
# dashboard.js against a fresh index.html breaks at the first id that changed.
# Without an explicit Cache-Control, browsers apply heuristic freshness to
# etag/last-modified and can skip revalidation entirely. "no-cache" means
# revalidate every time -- these are a few KB and answer 304.
NO_CACHE = {"Cache-Control": "no-cache"}
NO_STORE = {"Cache-Control": "no-store"}

REMOTE_TIMEOUT_S = 1.5

# The collectors keep delta state in module globals (network rates, vLLM
# windows), and several browsers -- or another node's board -- may poll at
# once. Serialise collection and share one sample per second between callers.
LOCAL_MIN_AGE_S = 1.0
_collect_lock = threading.Lock()
_last: tuple[float, dict] | None = None


def local_metrics() -> dict:
    # Imported here, not at module load: the collectors need SparkView and
    # NVIDIA hardware, and a board whose nodes are all remote (e.g. on a Mac)
    # has neither.
    from .collectors import get_metrics

    global _last
    with _collect_lock:
        if _last and time.monotonic() - _last[0] < LOCAL_MIN_AGE_S:
            return _last[1]
        data = get_metrics()
        _last = (time.monotonic(), data)
        return data


def remote_metrics(url: str) -> dict:
    with urllib.request.urlopen(url + "/api/metrics", timeout=REMOTE_TIMEOUT_S) as resp:
        return json.loads(resp.read())


async def node_payload(node: dict) -> dict:
    out = {"name": node["name"], "url": node["url"], "cards": node.get("cards")}
    try:
        if node["url"] == "local":
            data = await asyncio.to_thread(local_metrics)
        else:
            data = await asyncio.to_thread(remote_metrics, node["url"])
        out.update(ok=True, data=data)
    except Exception as err:
        out.update(ok=False, error=f"{type(err).__name__}: {err}")
    return out


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
    """This node only. Also what other boards fetch when they list us as a node."""
    data = await asyncio.to_thread(local_metrics)
    return JSONResponse(content=data, headers=NO_STORE)


@app.get("/api/board")
async def board():
    """Everything the page renders, in one request: config plus every node."""
    cfg = get_config()
    nodes = await asyncio.gather(*(node_payload(n) for n in cfg["nodes"]))
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
