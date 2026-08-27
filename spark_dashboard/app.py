"""SparkView Web Dashboard — FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .collectors import get_metrics

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SparkView Dashboard", docs_url=None, redoc_url=None)

# The frontend is three files that must stay in lockstep: a stale cached
# dashboard.js against a fresh index.html breaks at the first id that changed.
# Without an explicit Cache-Control, browsers apply heuristic freshness to
# etag/last-modified and can skip revalidation entirely. "no-cache" means
# revalidate every time -- these are a few KB and answer 304.
NO_CACHE = {"Cache-Control": "no-cache"}


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
    return FileResponse(
        STATIC / "index.html", media_type="text/html", headers=NO_CACHE
    )
