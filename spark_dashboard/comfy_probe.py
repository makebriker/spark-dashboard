"""ComfyUI detection and queue state.

Detection: every ``DETECT_TTL_S`` the process table is scanned for ComfyUI,
which runs as ``python main.py --listen ... [--port N]`` (default 8188). argv
alone can be just ``main.py``, so a Python process running main.py counts when
"comfy" appears in its interpreter path, script path or working directory.
Anything more exotic: list the URL under ``[comfyui] endpoints``.

Queue state comes from ``GET /queue`` and the last finished job from ``GET
/history?max_items=1``. Checked on 2026-11-07 against gb10-2's ComfyUI 0.33.0
source (b78cec87), and the idle answers of both nodes; no job was running, so
the /queue item shape is from source only: (number, prompt_id, graph,
extra_data, outputs). extra_data has only the submit time, so a job's elapsed
time counts from the first scrape that saw it running: up to one scrape short,
and it restarts if this dashboard does. /api/jobs is avoided because it walks
and normalises the whole history on every call.

Step progress exists only on ComfyUI's WebSocket, sent to the submitting client
if the job has a client_id and to every socket if not. ``_Progress`` listens as
one more client, so it sees jobs without a client_id; every job in both nodes'
history had none on 2026-11-07. A job with a client_id stays private on purpose:
reusing that id would replace the submitter's socket (server.py, /ws).

ComfyUI writes to its sockets one after another, so a client that stops reading
could slow it down. _Progress therefore has a thread that only reads, and takes
no lock the scrape loop holds. Its import is lazy: without the websockets package
a node loses step progress, not its metrics.
"""

from __future__ import annotations

import json
import math
import logging
import threading
import time
import urllib.request

import psutil

from .vllm_probe import _arg

log = logging.getLogger("spark_dashboard.comfy")

DETECT_TTL_S = 30.0
HTTP_TIMEOUT_S = 5.0
# /history is re-read whenever the running job changes, and on this cadence for
# jobs that came and went between two scrapes.
HISTORY_TTL_S = 30.0
# A timeout may just be a busy render; a refused connection means no server.
STALE_TIMEOUT_MAX_S = 120.0
STALE_MAX_S = 30.0
PROGRESS_RECONNECT_S = 5.0
DEFAULT_PORT = 8188
ERROR_MAX_CHARS = 160
# Loader inputs that name a job's main model. LoRA, CLIP and VAE are left out.
MODEL_INPUTS = ("unet_name", "ckpt_name")


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read()


def _is_comfy_server(cmd: list[str], cwd: str | None = None) -> bool:
    if not cmd[0].rsplit("/", 1)[-1].lower().startswith("python"):
        return False
    script = next((a for a in cmd[1:] if not a.startswith("-")), "")
    if script.rsplit("/", 1)[-1] != "main.py":
        return False
    return "comfy" in f"{cmd[0]} {script} {cwd or ''}".lower()


def _port_from_cmdline(cmd: list[str]) -> int:
    port = _arg(cmd, "--port")
    return int(port) if port else DEFAULT_PORT


def detect_endpoints() -> list[str]:
    found = set()
    for proc in psutil.process_iter(["cmdline", "cwd"]):
        cmd = proc.info.get("cmdline") or []
        if not cmd:
            continue
        try:
            if _is_comfy_server(cmd, proc.info.get("cwd")):
                found.add(f"http://127.0.0.1:{_port_from_cmdline(cmd)}")
        except ValueError:
            continue
    return sorted(found)


def job_model(graph) -> str | None:
    """Main model of a workflow graph, without its folder or extension."""
    if not isinstance(graph, dict):
        return None
    names = []
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        for key in MODEL_INPUTS:
            value = inputs.get(key) if isinstance(inputs, dict) else None
            if isinstance(value, str) and value:
                name = value.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if name not in names:
                    names.append(name)
    if not names:
        return None
    return names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")


def parse_last_job(history: dict) -> dict | None:
    """The newest /history entry: result, duration, end time, error."""
    if not history:
        return None
    item = list(history.values())[-1]
    status = item.get("status") or {}
    job = {
        "result": status.get("status_str") or "unknown",
        "duration_s": None,
        "ended_ms": None,
        "error": None,
    }
    started = None
    for entry in status.get("messages") or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        event, data = entry[0], entry[1]
        if event == "execution_start":
            started = data.get("timestamp")
        elif event in ("execution_success", "execution_error", "execution_interrupted"):
            job["ended_ms"] = data.get("timestamp")
            job["result"] = {"execution_success": "success", "execution_error": "error"}.get(event, "interrupted")
            if event == "execution_error":
                text = str(data.get("exception_message") or "").strip().split("\n")[0]
                where = data.get("node_type")
                job["error"] = ((f"{where}: " if where else "") + text)[:ERROR_MAX_CHARS]
    if started is not None and job["ended_ms"] is not None:
        job["duration_s"] = round((job["ended_ms"] - started) / 1000, 1)
    prompt = item.get("prompt")
    job["model"] = job_model(prompt[2]) if isinstance(prompt, (list, tuple)) and len(prompt) > 2 else None
    return job


def _finite(x) -> bool:
    # json.loads accepts NaN/Infinity, which JSONResponse then refuses.
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class _Progress:
    """Latest step of the running job, kept current by one WebSocket reader."""

    def __init__(self, url: str):
        self.ws_url = url.replace("http", "ws", 1) + "/ws"
        self.latest: dict | None = None
        self.last_error: str | None = None
        self.stop = threading.Event()
        self.ws = None
        self.thread: threading.Thread | None = None

    def feed(self, message: dict) -> None:
        kind, data = message.get("type"), message.get("data")
        if not isinstance(data, dict):
            return
        if kind == "progress":
            self.latest = data
        elif kind == "progress_state" and self.latest:
            # "executing" and "execution_success" are not sent for a job without
            # a client_id, so this is the only sign that the step's node ended.
            nodes = data.get("nodes")
            node = nodes.get(self.latest.get("node")) if isinstance(nodes, dict) else None
            if isinstance(node, dict) and node.get("state") != "running":
                self.latest = None

    def for_job(self, prompt_id: str, graph) -> dict | None:
        step = self.latest
        if not step or step.get("prompt_id") != prompt_id:
            return None
        value, total = step.get("value"), step.get("max")
        if not _finite(value) or not _finite(total) or total <= 0:
            return None
        node = graph.get(step.get("node")) if isinstance(graph, dict) else None
        return {"value": value, "max": total, "node_type": node.get("class_type") if isinstance(node, dict) else None}

    def _read(self, connect) -> None:
        with connect(self.ws_url, open_timeout=HTTP_TIMEOUT_S, close_timeout=1,
                     max_size=None, compression=None, proxy=None) as ws:
            self.ws = ws
            if self.stop.is_set():  # closed while connecting
                return
            if self.last_error:
                log.warning("%s recovered", self.ws_url)
                self.last_error = None
            for message in ws:
                if isinstance(message, str):  # bytes are preview images
                    self.feed(json.loads(message))

    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as err:
            log.warning("no ComfyUI step progress: %s", err)
            return
        while not self.stop.is_set():
            try:
                self._read(connect)
            except Exception as err:
                error = f"{type(err).__name__}: {err}"
                if error != self.last_error:
                    self.last_error = error
                    log.warning("%s failed: %s", self.ws_url, error)
            self.latest = None
            self.stop.wait(PROGRESS_RECONNECT_S)

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name=f"comfy-ws {self.ws_url}", daemon=True)
            self.thread.start()

    def close(self) -> None:
        self.stop.set()
        ws = self.ws
        if ws is not None:
            try:
                ws.close()
            except Exception as err:
                log.warning("%s close failed: %s", self.ws_url, err)


class _Endpoint:
    def __init__(self, url: str):
        self.url = url
        self.progress = _Progress(url)
        self.last_ok: tuple[float, dict] | None = None
        self.last_error: str | None = None
        self.job_id: str | None = None      # running job, as last seen
        self.job_seen = 0.0                 # when it was first seen running
        self.job_graph = None
        self.last_job: dict | None = None
        self.history_at: float | None = None

    def _refresh_history(self, now: float, job_changed: bool) -> None:
        if not job_changed and self.history_at is not None and now - self.history_at < HISTORY_TTL_S:
            return
        # A failed read keeps the last job and tries again on the next scrape.
        try:
            self.last_job = parse_last_job(json.loads(_http_get(self.url + "/history?max_items=1")))
            self.history_at = now
        except Exception as err:
            self.history_at = None
            log.warning("%s/history failed: %s: %s", self.url, type(err).__name__, err)

    def collect(self) -> dict:
        out: dict = {"endpoint": self.url, "ok": False}
        try:
            queue = json.loads(_http_get(self.url + "/queue"))
            running = queue["queue_running"]
            pending = queue["queue_pending"]
        except Exception as err:
            out["error"] = f"{type(err).__name__}: {err}"
            if out["error"] != self.last_error:
                self.last_error = out["error"]
                log.warning("%s/queue failed: %s", self.url, out["error"])
            timed_out = isinstance(err, TimeoutError) or isinstance(getattr(err, "reason", None), TimeoutError)
            grace = STALE_TIMEOUT_MAX_S if timed_out else STALE_MAX_S
            if self.last_ok and time.monotonic() - self.last_ok[0] < grace:
                # /queue times out during heavy renders; the reader still has fresh steps.
                cached = self.last_ok[1]
                step = self.progress.for_job(self.job_id, self.job_graph) if cached.get("running") else None
                return {**cached, "stale": True, "error": out["error"], "progress": step}
            return out
        now = time.monotonic()
        if self.last_error:
            log.warning("%s/queue recovered", self.url)
            self.last_error = None
        self.progress.start()

        current = running[0] if running else None
        job_id = current[1] if current else None
        extra = current[3] if current else None
        changed = job_id != self.job_id
        self.job_graph = current[2] if current else None
        if changed:
            self.job_id, self.job_seen = job_id, now
        self._refresh_history(now, changed)

        last = None
        if self.last_job:
            ended = self.last_job["ended_ms"]
            last = {k: v for k, v in self.last_job.items() if k != "ended_ms"}
            last["ago_s"] = max(0, round(time.time() - ended / 1000)) if ended is not None else None

        out.update(
            ok=True,
            running=len(running),
            pending=len(pending),
            model=job_model(current[2]) if current else None,
            job_elapsed_s=round(now - self.job_seen) if current else None,
            progress=self.progress.for_job(job_id, current[2]) if current else None,
            # ComfyUI treats client_id null as "no client" and broadcasts.
            progress_private=isinstance(extra, dict) and extra.get("client_id") is not None,
            last_job=last,
        )
        self.last_ok = (now, out)
        return out


_lock = threading.Lock()
_endpoints: dict[str, _Endpoint] = {}
_detected: list[str] = []
_detected_at = -DETECT_TTL_S


def _collect_all(cfg: dict) -> dict:
    """{"detected": bool, "servers": [...]} for this node, per the [comfyui] config."""
    global _detected, _detected_at
    with _lock:
        if not cfg["enabled"]:
            while _endpoints:
                _endpoints.popitem()[1].progress.close()
            return {"detected": False, "servers": []}
        urls = [u.rstrip("/") for u in cfg["endpoints"]]
        if not urls:
            now = time.monotonic()
            if now - _detected_at >= DETECT_TTL_S:
                _detected, _detected_at = detect_endpoints(), now
            urls = _detected
        for gone in _endpoints.keys() - set(urls):
            _endpoints.pop(gone).progress.close()
        servers = [_endpoints.setdefault(u, _Endpoint(u)).collect() for u in urls]
    return {"detected": any(s["ok"] for s in servers), "servers": servers}


# Its own loop: the vLLM sampler can block 30 s on a busy server.
_latest: dict = {"detected": False, "servers": []}
_sampler: threading.Thread | None = None


def get_comfy_info() -> dict:
    return _latest


def start_sampler(get_cfg) -> None:
    """Start the scrape loop once. get_cfg() returns the current [comfyui] config."""
    global _sampler
    if _sampler is not None:
        return

    def loop():
        global _latest
        while True:
            cfg = get_cfg()
            try:
                _latest = _collect_all(cfg)
            except Exception as err:  # never let the loop die
                _latest = {"detected": False, "servers": [], "error": str(err)}
            time.sleep(cfg["scrape_interval_s"])

    _sampler = threading.Thread(target=loop, name="comfy-sampler", daemon=True)
    _sampler.start()
