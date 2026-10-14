"""Board configuration — a TOML file, re-read whenever it changes on disk.

Lookup order (first that exists wins):

1. ``$SPARK_DASHBOARD_CONFIG``
2. ``config.toml`` in the working directory (the repo checkout under systemd)
3. ``~/.config/spark-dashboard/config.toml``

With no file at all the board behaves like v1: one local node, every card, and
the vLLM card shown only when a server is detected. ``config.toml`` is
gitignored, so ``deploy.sh``'s hard reset never touches it; start from
``config.example.toml``.
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import tomllib
from pathlib import Path

log = logging.getLogger("spark_dashboard.config")

MAX_NODES = 4

CARDS = ("vllm", "gpu", "memory", "cpu", "clock", "network", "disk")

VLLM_METRICS = (
    "decode_tps",     # aggregate generation throughput, tok/s
    "stream_tps",     # per-request decode speed, 1 / inter-token latency
    "prefill_tps",    # prompt tokens processed, tok/s
    "running",        # requests currently decoding
    "waiting",        # requests queued
    "kv_cache",       # KV cache utilisation, %
    "ttft",           # mean time to first token over the window
    "itl",            # mean time per output token (TPOT) over the window
    "e2e",            # mean end-to-end request latency over the window
    "req_rate",       # completed requests / s
    "prefix_hit",     # prefix-cache hit rate over the window
    "spec_accept",    # speculative-decoding acceptance rate over the window
    "spec_len",       # speculative decoding: mean tokens emitted per step
    "preemptions",    # preemptions over the window
    "max_model_len",  # served context length
)

# Token blocks in static/style.css; also the options of the header picker.
THEMES = ("classic", "refined", "quiet", "terminal", "light", "wall")

DEFAULTS: dict = {
    "board": {
        "title": "SPARKVIEW",
        "refresh_ms": 2000,
        # "columns": one column per node, cards stacked inside it.
        # "rows": one horizontal band per node, cards side by side.
        "layout": "columns",
        # Default look; the picker in the header overrides it per browser.
        "theme": "classic",
        "cards": list(CARDS),
    },
    "cards": {
        "gpu": {"processes": 5},
        "cpu": {"per_core": True},
        "memory": {"psi": True, "swap": True},
        "network": {"include": [], "exclude": [], "hide_down": False},
        "disk": {"include": [], "exclude": []},
    },
    "vllm": {
        # "auto": scan for a running vLLM server; true: same, but keep the card
        # visible (as "not detected") when none is found; false: never probe.
        "enabled": "auto",
        # Explicit base URLs, e.g. ["http://127.0.0.1:8000"]. Empty = detect
        # from the process table (`vllm serve ... --port N`).
        "endpoints": [],
        "metrics": [
            "decode_tps", "stream_tps", "prefill_tps", "running", "waiting",
            "kv_cache", "ttft", "itl", "prefix_hit",
        ],
        "rate_window_s": 10,
        # How often this node scrapes vLLM -- once, for every reader. A busy
        # server takes ~0.7 s to answer, so keep this well above that.
        "scrape_interval_s": 5,
        "sparkline": True,
        # "top": one full-width band above the node columns, gathering every
        # node's servers. "node": a card inside the column of the node serving.
        "placement": "top",
    },
    # Each node: name + url. "local" is collected in-process; anything else is
    # fetched from that node's own dashboard at <url>/api/metrics.
    "nodes": [{"name": "", "url": "local"}],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def _validate(cfg: dict) -> dict:
    board = cfg["board"]
    if not isinstance(board["cards"], list):
        raise TypeError(f"board.cards must be a list, got {board['cards']!r}")
    unknown = [c for c in board["cards"] if c not in CARDS]
    if unknown:
        log.warning("unknown cards ignored: %s (known: %s)", unknown, ", ".join(CARDS))
    board["cards"] = [c for c in board["cards"] if c in CARDS]
    if board["layout"] not in ("columns", "rows"):
        log.warning("board.layout must be 'columns' or 'rows', got %r", board["layout"])
        board["layout"] = "columns"
    if board["theme"] not in THEMES:
        log.warning("board.theme must be one of %s, got %r", ", ".join(THEMES), board["theme"])
        board["theme"] = "classic"
    board["refresh_ms"] = max(500, int(board["refresh_ms"]))

    vllm = cfg["vllm"]
    if not isinstance(vllm["metrics"], list) or not isinstance(vllm["endpoints"], list):
        raise TypeError("vllm.metrics and vllm.endpoints must be lists")
    unknown = [m for m in vllm["metrics"] if m not in VLLM_METRICS]
    if unknown:
        log.warning("unknown vllm.metrics ignored: %s", unknown)
    vllm["metrics"] = [m for m in vllm["metrics"] if m in VLLM_METRICS]
    if vllm["enabled"] not in ("auto", True, False):
        log.warning("vllm.enabled must be \"auto\", true or false, got %r", vllm["enabled"])
        vllm["enabled"] = "auto"
    vllm["rate_window_s"] = max(2, float(vllm["rate_window_s"]))
    vllm["scrape_interval_s"] = max(1, float(vllm["scrape_interval_s"]))
    # The window must span at least two scrapes to yield a rate.
    vllm["rate_window_s"] = max(vllm["rate_window_s"], 2 * vllm["scrape_interval_s"])
    if vllm["placement"] not in ("top", "node"):
        log.warning("vllm.placement must be 'top' or 'node', got %r", vllm["placement"])
        vllm["placement"] = "top"

    if not isinstance(cfg["nodes"], list):
        raise TypeError("nodes must be an array of tables ([[nodes]])")
    nodes = [n for n in cfg["nodes"] if isinstance(n, dict) and isinstance(n.get("url"), str) and n["url"]]
    dropped = [n for n in cfg["nodes"] if n not in nodes]
    if dropped:
        log.warning("nodes without a string url ignored: %s", dropped)
    if len(nodes) > MAX_NODES:
        log.warning("%d nodes configured; only the first %d are shown", len(nodes), MAX_NODES)
        nodes = nodes[:MAX_NODES]
    for n in nodes:
        n["url"] = n["url"].rstrip("/")
        n.setdefault("name", "")
        if "cards" in n:
            if not isinstance(n["cards"], list):
                raise TypeError(f"nodes.cards must be a list, got {n['cards']!r}")
            n["cards"] = [c for c in n["cards"] if c in CARDS]
    cfg["nodes"] = nodes or copy.deepcopy(DEFAULTS["nodes"])
    return cfg


def config_path() -> Path | None:
    candidates = []
    if os.environ.get("SPARK_DASHBOARD_CONFIG"):
        candidates.append(Path(os.environ["SPARK_DASHBOARD_CONFIG"]))
    candidates += [Path("config.toml"), Path.home() / ".config/spark-dashboard/config.toml"]
    return next((p for p in candidates if p.is_file()), None)


_lock = threading.Lock()
_cache: tuple[Path | None, float, dict] | None = None


def get_config() -> dict:
    """Current config. Cheap: one stat() per call, re-parses only on change.

    A file that fails to parse keeps the last good config rather than falling
    back to defaults, so a typo mid-edit doesn't flip the board around.
    """
    global _cache
    with _lock:
        try:
            path = config_path()
            mtime = path.stat().st_mtime if path else 0.0
        except OSError:  # removed between the lookup and the stat
            path, mtime = None, 0.0
        if _cache and _cache[0] == path and _cache[1] == mtime:
            return _cache[2]
        try:
            raw = tomllib.loads(path.read_text()) if path else {}
            cfg = _validate(_merge(DEFAULTS, raw))
        # Anything at all: a bad value must never escape into the sampler
        # threads, which have no other way to recover from a config edit.
        except Exception as err:
            log.error("config %s unusable, keeping previous: %s", path, err)
            cfg = _cache[2] if _cache else _validate(copy.deepcopy(DEFAULTS))
        else:
            if path:
                log.info("loaded config from %s", path)
        _cache = (path, mtime, cfg)
        return cfg
