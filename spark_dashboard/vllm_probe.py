"""vLLM detection and inference metrics.

A vLLM OpenAI-compatible server exposes Prometheus metrics at ``/metrics`` on
its API port. This module finds those servers, scrapes them, and turns the raw
counters and histograms into the numbers worth looking at: decode throughput,
per-stream decode speed, queue depth, KV-cache use, TTFT, inter-token latency.

Detection: every ``DETECT_TTL_S`` the process table is scanned for a vLLM API
server (``vllm serve ...`` or ``-m vllm.entrypoints.openai.api_server``) and
its ``--port`` is read off the command line (default 8000). Container processes
are visible in the host's process table, so a dockerised server is found too —
as long as the port is the same on the host (``--network host`` or ``-p N:N``).
Anything more exotic: list the URL under ``[vllm] endpoints``.

In a multi-node deployment (tensor-parallel over Ray) only the head node runs
the API server, so only the head node's board shows the vLLM card.

Rates are computed over a sliding window of samples kept here, not between
consecutive requests, so two browsers polling the same board don't halve each
other's window. Metric names changed across vLLM releases; each derived value
lists every spelling it accepts, newest first.

stdlib only — urllib, no prometheus_client, no requests.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections import deque

import psutil

DETECT_TTL_S = 30.0
MODELS_TTL_S = 60.0
# A busy vLLM API server answers /metrics (~70 KB) slowly -- 0.7 s measured
# on gb10 mid-generation, 1.3-5.3 s under Flash-Next TP=2 at 93 % GPU
# (2026-10-14), when 3.0 s reported a serving engine DOWN. This runs in the
# background sampler, never on a request path, so a generous timeout costs
# nothing.
HTTP_TIMEOUT_S = 10.0
# A failed scrape keeps showing the last good numbers, marked stale, for this
# long before the server is reported down. One slow answer is not an outage.
STALE_MAX_S = 30.0
DEFAULT_PORT = 8000

# Sample: name{label="v",...} value [timestamp]
_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# Gauges that describe a fraction/level: averaged across label sets (data-
# parallel engines each report their own). Everything else is summed.
_MEAN_GAUGES = {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:gpu_prefix_cache_hit_rate"}

_COUNTERS = {
    "gen_tokens": ("vllm:generation_tokens_total",),
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "requests": ("vllm:request_success_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_hits": ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    "prefix_queries": ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    "spec_accepted": ("vllm:spec_decode_num_accepted_tokens_total",),
    "spec_draft": ("vllm:spec_decode_num_draft_tokens_total",),
    "spec_drafts": ("vllm:spec_decode_num_drafts_total",),
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
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prefix_hit_v0": ("vllm:gpu_prefix_cache_hit_rate",),
}


def parse_prometheus(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Collapse exposition text to {metric: value}, merging label sets.

    Also returns the labels seen on the first vllm sample, which is where the
    served model name lives (``model_name="..."``).
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
        if lbl and not labels and name.startswith("vllm:"):
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
            if _is_vllm_server(cmd):
                url = f"http://127.0.0.1:{_port_from_cmdline(cmd)}"
                max_len = _arg(cmd, "--max-model-len")
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
            # Keep the window: a timeout is not a restart, and counter resets
            # are detected below when the server answers again.
            if self.last_ok and time.monotonic() - self.last_ok[0] < STALE_MAX_S:
                return {**self.last_ok[1], "stale": True, "error": out["error"]}
            return out
        now = time.monotonic()

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

        t0, prev = self._window(now, cur, window_s)
        # Counters only go down when the server restarted: start over.
        if any(cur.get(k, 0) < v for k, v in prev.items()):
            self.samples.clear()
            self.samples.append((now, cur))
            t0, prev = now, cur
        dt = now - t0

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
        itl = 1.0 / stream_tps if stream_tps else None
        prefix_hit = ratio("prefix_hits", "prefix_queries")
        if prefix_hit is None and gauges["prefix_hit_v0"] is not None:
            prefix_hit = gauges["prefix_hit_v0"]
        model = self._models()

        out.update(
            ok=True,
            model=labels.get("model_name") or (model or {}).get("id"),
            window_s=round(dt, 1),
            metrics={
                "decode_tps": rate("gen_tokens"),
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
