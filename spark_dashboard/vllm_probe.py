"""vLLM and TensorFold detection and inference metrics.

A vLLM OpenAI-compatible server exposes Prometheus metrics at ``/metrics`` on
its API port. A TensorFold server (``tensorfold serve``) does the same with
``tensorfold_*`` names; it is detected alongside vLLM and has no histograms
and no KV-cache gauge, so kv_cache / ttft / e2e stay empty for it. Its /metrics
counters move only when a request finishes, so the decode rate comes live from
its ``/health`` instead, and per-stream speed and prefix hit are held from the
last completed request(s). This module finds those servers, scrapes them, and turns the raw
counters and histograms into the numbers worth looking at: decode throughput,
per-stream decode speed, queue depth, KV-cache use, TTFT, inter-token latency.

Detection: every ``DETECT_TTL_S`` the process table is scanned for a vLLM API
server (``vllm serve ...`` or ``-m vllm.entrypoints.openai.api_server``) or a
TensorFold server (``python -m tensorfold serve ...`` or ``tensorfold serve
...``, rank 0 only) and its ``--port`` is read off the command line (default
8000). Container processes
are visible in the host's process table, so a dockerised server is found too —
as long as the port is the same on the host (``--network host`` or ``-p N:N``).
Anything more exotic: list the URL under ``[vllm] endpoints``.

In a multi-node deployment (tensor-parallel over Ray, or TensorFold ``--rank``
> 0) only the head node runs the API server, so only the head node's board shows the vLLM card.

Rates are computed over a sliding window of samples kept here, not between
consecutive requests, so two browsers polling the same board don't halve each
other's window. Metric names changed across vLLM releases; each derived value
lists every spelling it accepts, newest first.

stdlib only — urllib, no prometheus_client, no requests.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from collections import deque

import psutil

log = logging.getLogger("spark_dashboard.vllm")

DETECT_TTL_S = 30.0
MODELS_TTL_S = 60.0
# vLLM's API server stops answering /metrics while it streams a request:
# measured on gb10 (TP=2, 2026-10-14) one call blocked 20 s and returned the
# instant the request finished, then took 3 ms. So a scrape takes as long as
# the longest in-flight request. This runs in the background sampler, never
# on a request path, so a long timeout costs nothing.
HTTP_TIMEOUT_S = 30.0
# After a failed scrape the last good numbers stay up, marked stale, before
# the server is reported down. A timeout is a busy server (see above), so it
# gets a long grace; a refused connection means no server is listening.
STALE_TIMEOUT_MAX_S = 120.0
STALE_MAX_S = 30.0
DEFAULT_PORT = 8000
# Fewest generated tokens a TensorFold completion needs to update per-stream speed.
TF_MIN_STREAM_TOKENS = 64

# Sample: name{label="v",...} value [timestamp]
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# Gauges that describe a fraction/level: averaged across label sets (data-
# parallel engines each report their own). Everything else is summed.
_MEAN_GAUGES = {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate"}

_COUNTERS = {
    "gen_tokens": ("vllm:generation_tokens_total", "tensorfold_completion_tokens_total"),
    "prompt_tokens": ("vllm:prompt_tokens_total", "tensorfold_prompt_tokens_total"),
    "requests": ("vllm:request_success_total", "tensorfold_requests_total"),
    "preemptions": ("vllm:num_preemptions_total",),
    # TensorFold: cached prompt tokens over prompt tokens (vLLM's queries
    # counter is in prompt tokens too).
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits", "tensorfold_cached_tokens_total"),
    "prefix_queries": (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
        "tensorfold_prompt_tokens_total",
    ),
    "spec_accepted": ("vllm:spec_decode_num_accepted_tokens_total",),
    "spec_draft": ("vllm:spec_decode_num_draft_tokens_total",),
    "spec_drafts": ("vllm:spec_decode_num_drafts_total",),
    # TensorFold: decode-seconds summed over requests, so completion tokens per
    # decode-second is per-stream speed (verified 2026-10-19: 600 tokens / 13.2 s =
    # 45 tok/s). decode_rounds_total is NOT tokens (17463 rounds for those 600).
    # All tensorfold_* counters move only when a request completes, so on that
    # server every rate here jumps at completion instead of streaming.
    "decode_seconds": ("tensorfold_decode_seconds_total",),
}

_HISTOGRAMS = {
    "ttft": ("vllm:time_to_first_token_seconds",),
    # Observed once per engine *step* per running request -- not per token.
    # With speculative decoding a step emits several tokens, so sum/count is
    # the step time and 1/that undercounts decode speed by the acceptance
    # length (3x on an MTP-3 server). Use it as stream-time, not as latency.
    "itl": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
    # One observation per finished request: its decode time / output tokens.
    # Correct per token, but only moves when a request completes.
    "tpot_req": ("vllm:request_time_per_output_token_seconds",),
    "e2e": ("vllm:e2e_request_latency_seconds",),
}

_GAUGES = {
    "running": ("vllm:num_requests_running", "tensorfold_requests_inflight"),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prefix_hit_v0": ("vllm:gpu_prefix_cache_hit_rate",),
}


def parse_prometheus(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Collapse exposition text to {metric: value}, merging label sets.

    Also returns the labels seen on the first vllm / tensorfold sample, which is
    where the served model name lives (``model_name="..."`` for vLLM,
    ``model="..."`` for TensorFold).
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, lbl, raw = m.groups()
        if name.endswith("_created") or "_bucket" in name:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val != val:  # NaN
            continue
        if lbl and not labels and name.startswith(("vllm:", "tensorfold_")):
            labels = dict(_LABEL.findall(lbl))
        sums[name] = sums.get(name, 0.0) + val
        counts[name] = counts.get(name, 0) + 1
    for name in _MEAN_GAUGES & sums.keys():
        sums[name] /= counts[name]
    return sums, labels


def _first(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in values:
            return values[n]
    return None


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "text/plain, application/json"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read()


def _arg(cmd: list[str], flag: str) -> str | None:
    for i, arg in enumerate(cmd):
        if arg == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _port_from_cmdline(cmd: list[str]) -> int:
    port = _arg(cmd, "--port")
    return int(port) if port else DEFAULT_PORT


def _is_tensorfold_server(cmd: list[str]) -> bool:
    # `serve --rank N` runs on every node; only rank 0 has the API port.
    rank = _arg(cmd, "--rank")
    if rank is not None and rank != "0":
        return False
    for i, arg in enumerate(cmd[:-1]):
        if cmd[i + 1] != "serve":
            continue
        if arg.rsplit("/", 1)[-1] == "tensorfold":
            return True
        if arg == "tensorfold" and i > 0 and cmd[i - 1] == "-m":
            return True
    return False


def _is_vllm_server(cmd: list[str]) -> bool:
    # Multi-node (--nnodes N): ranks > 0 run `vllm serve ... --headless` and
    # open no API port. Their numbers are reported by the rank-0 node.
    if "--headless" in cmd:
        return False
    joined = " ".join(cmd)
    if "vllm.entrypoints.openai.api_server" in joined:
        return True
    # `vllm serve <model>` — argv[0] may be a full path to the vllm script, or
    # python running it.
    for i, arg in enumerate(cmd[:-1]):
        if arg.rsplit("/", 1)[-1] == "vllm" and cmd[i + 1] == "serve":
            return True
    return False


# url -> max_model_len read off the command line. /v1/models carries it too,
# but sits behind --api-key while /metrics does not.
_cmdline_max_len: dict[str, int] = {}


def detect_endpoints() -> list[str]:
    found: dict[str, int | None] = {}
    for proc in psutil.process_iter(["cmdline"]):
        cmd = proc.info.get("cmdline") or []
        if not cmd:
            continue
        try:
            vllm, tf = _is_vllm_server(cmd), _is_tensorfold_server(cmd)
            if vllm or tf:
                url = f"http://127.0.0.1:{_port_from_cmdline(cmd)}"
                max_len = _arg(cmd, "--max-model-len" if vllm else "--context")
                found[url] = int(max_len) if max_len and max_len.isdigit() else found.get(url)
        except ValueError:
            continue
    _cmdline_max_len.clear()
    _cmdline_max_len.update({u: n for u, n in found.items() if n})
    return sorted(found)


class _Endpoint:
    def __init__(self, url: str):
        self.url = url
        self.samples: deque[tuple[float, dict[str, float]]] = deque()
        self.model: dict | None = None
        self.model_at = 0.0
        self.last_ok: tuple[float, dict] | None = None
        self.last_error: str | None = None
        # TensorFold: /metrics snapshot at the last completion, the per-request
        # figures computed from it, and when that completion was seen.
        self.done_base: dict[str, float] | None = None
        self.done_stats: dict[str, float | None] = {}
        self.done_at: float | None = None

    def _models(self) -> dict | None:
        """/v1/models, cached. 401 under --api-key is fine: the model name
        also comes from the metric labels, only max_model_len is lost."""
        now = time.monotonic()
        if now - self.model_at < MODELS_TTL_S:
            return self.model
        self.model_at = now
        try:
            data = json.loads(_http_get(self.url + "/v1/models"))["data"][0]
            self.model = {"id": data.get("id"), "max_model_len": data.get("max_model_len")}
        except Exception:
            self.model = None
        return self.model

    def _window(self, now: float, values: dict[str, float], window_s: float):
        """Append this sample; return the oldest one still inside the window."""
        self.samples.append((now, values))
        while len(self.samples) > 2 and now - self.samples[1][0] >= window_s:
            self.samples.popleft()
        return self.samples[0]

    def collect(self, window_s: float) -> dict:
        out: dict = {"endpoint": self.url, "ok": False}
        try:
            raw, labels = parse_prometheus(_http_get(self.url + "/metrics").decode())
        except Exception as err:
            out["error"] = f"{type(err).__name__}: {err}"
            # One line per distinct failure, and one on recovery, so a DOWN
            # on the board can be matched to its cause in the journal.
            if out["error"] != self.last_error:
                self.last_error = out["error"]
                log.warning("%s/metrics scrape failed: %s", self.url, out["error"])
            # Keep the window: a timeout is not a restart, and counter resets
            # are detected below when the server answers again.
            timed_out = isinstance(err, TimeoutError) or isinstance(getattr(err, "reason", None), TimeoutError)
            grace = STALE_TIMEOUT_MAX_S if timed_out else STALE_MAX_S
            if self.last_ok and time.monotonic() - self.last_ok[0] < grace:
                return {**self.last_ok[1], "stale": True, "error": out["error"]}
            return out
        now = time.monotonic()
        if self.last_error:
            log.warning("%s/metrics scrape recovered", self.url)
            self.last_error = None

        cur: dict[str, float] = {}
        for key, names in _COUNTERS.items():
            v = _first(raw, names)
            if v is not None:
                cur[key] = v
        for key, names in _HISTOGRAMS.items():
            s = _first(raw, tuple(n + "_sum" for n in names))
            c = _first(raw, tuple(n + "_count" for n in names))
            if s is not None and c is not None:
                cur[key + "_sum"], cur[key + "_count"] = s, c

        # TensorFold bumps every /metrics counter only when a request finishes,
        # and an agent turn decodes for minutes, so a 10 s window of /metrics
        # alone reads 0 tok/s while it serves. /health counts the tokens of
        # requests still running, so the live decode rate comes from there. A
        # failed /health must never fail the scrape.
        tf = "decode_seconds" in cur
        streams = None
        if tf:
            try:
                health = json.loads(_http_get(self.url + "/health"))
                cur["gen_live"] = float(health["completion_tokens_total"])
                streams = health.get("streams")
                running_now = health.get("requests_running")
            except Exception:
                streams = running_now = None

        t0, prev = self._window(now, cur, window_s)
        # Counters only go down when the server restarted: start over. A scrape
        # whose /health failed has no gen_live; that is not a reset.
        reset = any(cur.get(k, 0) < v for k, v in prev.items() if k in cur or k != "gen_live")
        if reset:
            self.samples.clear()
            self.samples.append((now, cur))
            t0, prev = now, cur
        dt = now - t0

        if tf:
            if reset:
                self.done_base, self.done_stats, self.done_at = cur, {}, None
            elif self.done_base is None:
                self.done_base = cur
            elif cur.get("requests") != self.done_base.get("requests"):
                def d(key):
                    if key in cur and key in self.done_base:
                        return cur[key] - self.done_base[key]
                    return 0.0
                d_gen, d_dec, d_hits, d_queries = d("gen_tokens"), d("decode_seconds"), d("prefix_hits"), d("prefix_queries")
                # TensorFold books prefill time as decode time (its prefill
                # counter stays 0), so a short reply after a long prompt reads
                # as ~1 tok/s. Per-stream speed keeps its last value until a
                # completion has enough tokens to swamp that.
                stream_tps = self.done_stats.get("stream_tps")
                if d_gen >= TF_MIN_STREAM_TOKENS and d_dec > 0:
                    stream_tps = d_gen / d_dec
                self.done_stats = {
                    "stream_tps": stream_tps,
                    "prefix_hit": d_hits / d_queries if d_queries > 0 else None,
                }
                self.done_base, self.done_at = cur, now

        def delta(key):
            if key in cur and key in prev:
                return cur[key] - prev[key]
            return None

        def rate(key):
            d = delta(key)
            return d / dt if d is not None and dt >= 0.5 else None

        def ratio(num, den):
            n, d = delta(num), delta(den)
            return n / d if n is not None and d else None

        gauges = {k: _first(raw, names) for k, names in _GAUGES.items()}
        # Per-stream decode speed = decode tokens / stream-seconds spent
        # decoding. The ITL sum is exactly those stream-seconds (each running
        # request contributes its own step intervals), so this holds for any
        # concurrency and any speculative acceptance length. First tokens are
        # timed by TTFT, not ITL, so they come out of the numerator.
        stream_tps = None
        gen, itl_time = delta("gen_tokens"), delta("itl_sum")
        if gen is not None and itl_time:
            stream_tps = max(0.0, gen - (delta("ttft_count") or 0)) / itl_time or None
        elif (tpot := ratio("tpot_req_sum", "tpot_req_count")):
            stream_tps = 1.0 / tpot
        itl = None
        prefix_hit = ratio("prefix_hits", "prefix_queries")
        if prefix_hit is None and gauges["prefix_hit_v0"] is not None:
            prefix_hit = gauges["prefix_hit_v0"]
        decode_tps = rate("gen_tokens")
        if tf:
            # Held from the last completion: a window of counters that move
            # only at completion would be empty most of the time.
            stream_tps = self.done_stats.get("stream_tps")
            prefix_hit = self.done_stats.get("prefix_hit")
            live = rate("gen_live")
            if live is not None:
                decode_tps = live
            if streams:
                busy = (streams.get("decoding") or 0) + (streams.get("prefilling") or 0)
                gauges["running"] = busy
                if running_now is not None:
                    gauges["waiting"] = max(0, running_now - busy)
        if stream_tps:
            itl = 1.0 / stream_tps
        model = self._models()

        out.update(
            ok=True,
            model=labels.get("model_name") or labels.get("model") or (model or {}).get("id"),
            # Identifies this sample: the board pushes one sparkline point per
            # new value, not per browser tick.
            sample_at=round(time.time(), 1),
            window_s=round(dt, 1),
            metrics={
                "decode_tps": decode_tps,
                "stream_tps": stream_tps,
                "prefill_tps": rate("prompt_tokens"),
                "running": gauges["running"],
                "waiting": gauges["waiting"],
                "kv_cache": gauges["kv_cache"] * 100 if gauges["kv_cache"] is not None else None,
                "ttft": ratio("ttft_sum", "ttft_count"),
                "itl": itl,
                "e2e": ratio("e2e_sum", "e2e_count"),
                "req_rate": rate("requests"),
                "prefix_hit": prefix_hit * 100 if prefix_hit is not None else None,
                "spec_accept": (r * 100 if (r := ratio("spec_accepted", "spec_draft")) is not None else None),
                "spec_len": (1 + r if (r := ratio("spec_accepted", "spec_drafts")) is not None else None),
                "preemptions": delta("preemptions"),
                "max_model_len": (model or {}).get("max_model_len") or _cmdline_max_len.get(self.url),
            },
        )
        if tf:
            out["last_done_s"] = round(now - self.done_at, 0) if self.done_at is not None else None
            out["per_request_from"] = "last_completion"
        self.last_ok = (now, out)
        return out


_lock = threading.Lock()
_endpoints: dict[str, _Endpoint] = {}
_detected: list[str] = []
_detected_at = -DETECT_TTL_S


def _collect_all(cfg: dict) -> dict:
    """{"detected": bool, "servers": [...]} for this node, per the [vllm] config."""
    global _detected, _detected_at
    if cfg["enabled"] is False:
        return {"detected": False, "servers": []}
    with _lock:
        urls = [u.rstrip("/") for u in cfg["endpoints"]]
        if not urls:
            now = time.monotonic()
            if now - _detected_at >= DETECT_TTL_S:
                _detected, _detected_at = detect_endpoints(), now
            urls = _detected
        for gone in _endpoints.keys() - set(urls):
            del _endpoints[gone]
        servers = [_endpoints.setdefault(u, _Endpoint(u)).collect(cfg["rate_window_s"]) for u in urls]
    return {"detected": any(s["ok"] for s in servers), "servers": servers}


# ── Background sampler ────────────────────────────────────────────────────────
# vLLM is scraped by exactly one loop per node, every scrape_interval_s, no
# matter how many browsers, boards or pollers read this node. Readers get the
# latest result from get_vllm_info() without touching vLLM at all.

_latest: dict = {"detected": False, "servers": []}
_sampler: threading.Thread | None = None


def get_vllm_info(_cfg: dict | None = None) -> dict:
    return _latest


def start_sampler(get_cfg) -> None:
    """Start the scrape loop once. get_cfg() returns the current [vllm] config."""
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

    _sampler = threading.Thread(target=loop, name="vllm-sampler", daemon=True)
    _sampler.start()
