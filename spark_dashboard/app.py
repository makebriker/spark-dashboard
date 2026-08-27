"""SparkView Web Dashboard — FastAPI application."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from .collectors import get_metrics

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SparkView Dashboard", docs_url=None, redoc_url=None)

# The frontend is three files that must stay in lockstep: a stale cached
# dashboard.js against a fresh index.html breaks at the first id that changed.
# Without an explicit Cache-Control, browsers apply heuristic freshness to
# etag/last-modified and can skip revalidation entirely. "no-cache" means
# revalidate every time -- these are a few KB and answer 304.
NO_CACHE = {"Cache-Control": "no-cache"}


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
    data = get_metrics()
    return JSONResponse(content=data, headers={"Cache-Control": "no-store"})


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
