"""SparkView Web Dashboard — FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .collectors import get_metrics

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SparkView Dashboard", docs_url=None, redoc_url=None)


@app.get("/api/metrics")
async def metrics():
    data = get_metrics()
    return JSONResponse(content=data, headers={"Cache-Control": "no-store"})


@app.get("/style.css")
async def style():
    return FileResponse(STATIC / "style.css", media_type="text/css")


@app.get("/dashboard.js")
async def javascript():
    return FileResponse(
        STATIC / "dashboard.js", media_type="application/javascript"
    )


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")