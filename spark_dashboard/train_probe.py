"""Training runs: ai-toolkit LoRA training, read from its own log.

Detection: every ``DETECT_TTL_S`` the process table is scanned for ai-toolkit,
which runs as ``python run.py <config>.yaml`` from its checkout (a folder with a
``toolkit`` package beside run.py). Its worker processes share that command
line, so a match whose parent also matches is skipped, and stays skipped: a
worker outliving its run is reparented to init and would otherwise read as a
new run (seen when replaying a log on gb10-2, 2026-11-08).

Everything else comes from the file the run's stdout points to, through
``/proc/<pid>/fd/1``. tqdm writes to stderr, so this needs ``> log 2>&1``. No
path needs configuring. The log is read from the start (``READ_MAX`` per scrape),
then only the new bytes; a later ``Running 1 job`` line (a run appended with
``>>``) starts the reading over. Its shape, checked 2026-11-07 against gb10-2's ai-toolkit logs
(train_v03p.log, a finished 1000-step run, and train_v04p.log, running):

- ``Running 1 job`` then the process config as indented JSON: steps,
  save_every, sample_every, training_folder.
- tqdm bars written with ``\\r``. The training bar is ``<name>: 7%|..| 71/1000
  [25:00<4:04:54, 15.82s/it, lr: 1.0e-04 loss: 4.216e-02]``; other bars are
  ``Caching latents to disk``, ``Caching text embeddings to disk``, ``Loading
  weights`` and ``Generating Samples`` (one tick per sample prompt).
- ``Saving at step N``, ``Saved checkpoint to <path>``; after the last step a
  final checkpoint without a step suffix, then one more round of samples.
- A crash ends in a Python traceback.

The rate is measured from the training bar's own elapsed time between step
increments, leaving out any interval that held sampling or saving. tqdm's s/it
is a smoothed average and runs 70% high right after a sample round.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import threading
import time

import psutil

log = logging.getLogger("spark_dashboard.train")

DETECT_TTL_S = 15.0
# A finished or failed run stays on the board this long, or until the next one.
FINISHED_KEEP_S = 6 * 3600.0
RATE_WINDOW = 20           # step intervals averaged for s/step
LOSS_SMOOTH = 20           # trailing mean for the loss tile and curve
LOSS_POINTS = 120          # points sent for the loss curve
READ_MAX = 8 << 20         # never read more than this from one log at once
ERROR_MAX_CHARS = 160

_BAR = re.compile(
    r"^(?P<name>[^:\r\n]{1,80}?):\s+\d+%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)(?:<[^,\]]*)?(?P<post>[^\]]*)\]")
_LOSS = re.compile(r"loss:\s*([-+0-9.eE]+)")
# print() output is block-buffered into a file, so these may share a line with a bar.
_SAVED = re.compile(r"Saved checkpoint to (?P<path>.+?\.safetensors)")
_SAVING = re.compile(r"Saving at step \d+")
_NEW_JOB = re.compile(r"^Running \d+ jobs?$")
_STEP_SUFFIX = re.compile(r"_(\d{6,})\.safetensors$")
PHASES = (
    ("Generating Samples", "SAMPLING"),
    ("Caching", "CACHING"),
    ("Loading", "LOADING"),
)


def _seconds(text: str) -> int:
    total = 0
    for part in text.split(":"):
        total = total * 60 + int(part or 0)
    return total


def _is_toolkit_run(cmd: list[str], cwd: str | None) -> str | None:
    """The config path if this is ``python run.py <config>`` in an ai-toolkit checkout."""
    if len(cmd) < 3 or not os.path.basename(cmd[0]).startswith("python"):
        return None
    args = [a for a in cmd[1:] if not a.startswith("-")]
    if len(args) < 2 or os.path.basename(args[0]) != "run.py":
        return None
    if not args[1].endswith((".yaml", ".yml", ".json")):
        return None
    root = os.path.dirname(os.path.join(cwd or "", args[0]))
    if not os.path.isdir(os.path.join(root, "toolkit")):
        return None
    return args[1]


_workers: set[int] = set()


def detect_runs() -> dict[int, dict]:
    """{pid: {"config", "log", "started"}} for running ai-toolkit trainings."""
    found: dict[int, dict] = {}
    procs = {}
    for p in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        try:
            cmd = p.info["cmdline"] or []
            if "run.py" not in " ".join(cmd):
                continue
            config = _is_toolkit_run(cmd, p.cwd())
        except (psutil.Error, OSError):
            continue
        if config:
            procs[p.info["pid"]] = (p.info["ppid"], cmd, config, p.info["create_time"])
    _workers.intersection_update(procs)
    for pid, (ppid, cmd, config, started) in procs.items():
        if ppid in procs and procs[ppid][1] == cmd:
            _workers.add(pid)  # a worker of the run above it
        if pid in _workers:
            continue
        try:
            target = os.readlink(f"/proc/{pid}/fd/1")
        except OSError:
            target = None
        found[pid] = {"config": config, "log": target if target and os.path.isfile(target) else None,
                      "started": started}
    return found


class _Run:
    """One training run: its log read incrementally, and what it says."""

    def __init__(self, pid: int, config: str, log_path: str | None, started: float):
        self.pid = pid
        self.config_path = config
        self.log_path = log_path
        self.started = started
        self.ended: float | None = None
        self.log_error: str | None = None
        self.offset = 0
        self.partial = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._reset()

    def _reset(self) -> None:
        """Forget what the log said: it was truncated, or a new job was appended to it."""
        self.head = ""                  # text up to the config JSON
        self.cfg: dict | None = None
        self.name = os.path.splitext(os.path.basename(self.config_path))[0]
        self.phase = "STARTING"
        self.phase_bar: dict | None = None
        self.step = 0
        self.total: int | None = None
        self.last_elapsed: int | None = None
        self.intervals: list[tuple[int, int]] = []   # (steps, seconds), clean only
        self.paused = False             # an interval held sampling or saving
        self.losses: list[float] = []
        self.sample_s: int | None = None
        self.checkpoint: str | None = None
        self.tail: list[str] = []       # last non-bar lines, for a crash message
        self.traceback = False          # since the last training progress

    # ---------- log reading ----------

    def read(self) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if size < self.offset:      # truncated or replaced: start over
                    self.offset, self.partial = 0, ""
                    self.decoder.reset()
                    self._reset()
                f.seek(self.offset)
                chunk = f.read(READ_MAX)
        except OSError as err:
            self.log_error = f"{type(err).__name__}: {err}"
            return
        self.log_error = None
        self.offset += len(chunk)
        text = self.partial + self.decoder.decode(chunk)
        pieces = re.split(r"[\r\n]", text)
        self.partial = pieces.pop()
        for line in pieces:
            self._line(line)
        # tqdm writes "\r" + bar, so the newest bar has no terminator yet. Bars
        # parse idempotently, so read it now and again once it is complete.
        m = _BAR.match(self.partial)
        if m:
            self._bar(m)

    def _line(self, line: str) -> None:
        if _NEW_JOB.match(line.strip()):
            self._reset()               # a run appended to an old log (">>")
        if self.cfg is None:
            self._find_config(line)
        m = _BAR.match(line)
        if m:
            self._bar(m)
            rest = line[m.end():].strip()   # buffered print() output after the bar
            if rest:
                self._line(rest)
            return
        stripped = line.strip()
        if not stripped:
            return
        if _SAVING.search(stripped):
            self.phase, self.phase_bar, self.paused = "SAVING", None, True
        saved = _SAVED.search(stripped)
        if saved:
            self.checkpoint = saved["path"]
        if stripped.startswith("Traceback (most recent call last)"):
            self.traceback = True
        self.tail = (self.tail + [stripped])[-3:]

    def _bar(self, m: re.Match) -> None:
        loss = _LOSS.search(m["post"])
        if loss or (self.cfg is not None and m["name"] == self.name):
            self._train_line(m, loss)
            return
        self.phase = next((p for prefix, p in PHASES if m["name"].startswith(prefix)), "WORKING")
        self.phase_bar = {"label": m["name"], "value": int(m["n"]), "max": int(m["total"])}
        self.paused = True
        if self.phase == "SAMPLING" and m["n"] == m["total"]:
            self.sample_s = _seconds(m["elapsed"])

    def _find_config(self, line: str) -> None:
        # The process config is printed as JSON after "Running 1 job".
        if len(self.head) > 64 << 10:
            return
        self.head += line + "\n"
        if line != "}":
            return
        start = self.head.rfind("\n{\n")
        if start < 0:
            return
        try:
            self.cfg = json.loads(self.head[start + 1:])
        except ValueError:
            self.cfg = None

    def _train_line(self, m: re.Match, loss: re.Match | None) -> None:
        n, total, elapsed = int(m["n"]), int(m["total"]), _seconds(m["elapsed"])
        if self.name == os.path.splitext(os.path.basename(self.config_path))[0]:
            self.name = m["name"].strip()
        self.total = total
        self.phase, self.phase_bar = "TRAINING", None
        self.traceback = False          # an earlier traceback was not fatal
        # Intervals run from one increment to the next; the first increment
        # only sets the mark (the line before it carries the same elapsed).
        if n > self.step:
            if self.last_elapsed is not None and not self.paused and elapsed >= self.last_elapsed:
                self.intervals = (self.intervals + [(n - self.step, elapsed - self.last_elapsed)])[-RATE_WINDOW:]
            if loss:
                try:
                    self.losses.append(float(loss[1]))
                except ValueError:
                    pass
            self.paused = False
            self.last_elapsed = elapsed
        elif n < self.step:     # the bar restarted
            self.last_elapsed = None
        self.step = n

    # ---------- summary ----------

    def _every(self, section: str, key: str) -> int | None:
        value = ((self.cfg or {}).get(section) or {}).get(key)
        return value if isinstance(value, int) and value > 0 else None

    def _sample_rounds_left(self) -> int:
        every = self._every("sample", "sample_every")
        if self.total is None or not (self.cfg or {}).get("sample"):
            return 0
        rounds = 1  # ai-toolkit samples once more after the last step
        if every:
            rounds += len(range((self.step // every + 1) * every, self.total, every))
        return rounds

    def _next(self, section: str, key: str) -> int | None:
        every = self._every(section, key)
        if not every or self.total is None:
            return None
        nxt = (self.step // every + 1) * every
        return nxt if nxt < self.total else self.total

    def summary(self, alive: bool) -> dict:
        steps = sum(s for s, _ in self.intervals)
        rate = sum(t for _, t in self.intervals) / steps if steps else None
        eta = None
        if alive and rate is not None and self.total is not None:
            left = max(0, self.total - self.step) * rate
            eta = left + self._sample_rounds_left() * (self.sample_s or 0)

        smooth = []
        for i in range(len(self.losses)):
            window = self.losses[max(0, i - LOSS_SMOOTH + 1):i + 1]
            smooth.append(sum(window) / len(window))
        if len(smooth) > LOSS_POINTS:
            stride = len(smooth) / LOSS_POINTS
            smooth = [smooth[int(i * stride)] for i in range(LOSS_POINTS - 1)] + [smooth[-1]]

        checkpoint = None
        if self.checkpoint:
            suffix = _STEP_SUFFIX.search(self.checkpoint)
            try:
                age = max(0, round(time.time() - os.stat(self.checkpoint).st_mtime))
            except OSError:
                age = None
            checkpoint = {"step": int(suffix[1]) if suffix else self.total, "final": not suffix, "ago_s": age}

        samples = None
        folder = (self.cfg or {}).get("training_folder")
        if folder:
            try:
                files = [e for e in os.scandir(os.path.join(folder, self.name, "samples")) if e.is_file()]
                newest = max((e.stat().st_mtime for e in files), default=None)
                samples = {"count": len(files),
                           "ago_s": max(0, round(time.time() - newest)) if newest else None}
            except OSError:
                pass

        state = self.phase
        error = None
        if not alive:
            if self.traceback and self.tail and self.tail[-1].startswith("KeyboardInterrupt"):
                state = "STOPPED"
            elif self.traceback:
                state = "FAILED"
                error = (self.tail[-1] if self.tail else "traceback in the log")[:ERROR_MAX_CHARS]
            elif self.total is not None and self.step >= self.total:
                state = "DONE"
            else:
                state = "STOPPED"

        return {
            "pid": self.pid,
            "name": self.name,
            "config": self.config_path,
            "log": self.log_path,
            "log_error": self.log_error if self.log_path else "stdout is not a file",
            "state": state,
            "phase_bar": self.phase_bar if alive else None,
            "step": self.step,
            "total": self.total,
            "s_per_step": round(rate, 2) if rate is not None else None,
            "eta_s": round(eta) if eta is not None else None,
            "running_s": round((self.ended or time.time()) - self.started),
            "ended_ago_s": round(time.time() - self.ended) if self.ended else None,
            "loss": round(smooth[-1], 5) if smooth else None,
            "loss_last": self.losses[-1] if self.losses else None,
            "loss_points": [round(v, 5) for v in smooth],
            "next_sample": self._next("sample", "sample_every") if alive else None,
            "next_save": self._next("save", "save_every") if alive else None,
            "sample_s": self.sample_s,
            "checkpoint": checkpoint,
            "samples": samples,
            "error": error,
        }


_lock = threading.Lock()
_runs: dict[int, _Run] = {}
_detected: dict[int, dict] = {}
_detected_at = -DETECT_TTL_S


def _alive(pid: int, started: float | None = None) -> bool:
    try:
        proc = psutil.Process(pid)
        if started is not None and abs(proc.create_time() - started) > 1:
            return False                # the pid was reused
        return proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _collect_all(cfg: dict) -> dict:
    """{"detected": bool, "runs": [...]} for this node, per the [training] config."""
    global _detected, _detected_at
    with _lock:
        if not cfg["enabled"]:
            _runs.clear()
            return {"detected": False, "runs": []}
        now = time.monotonic()
        if now - _detected_at >= DETECT_TTL_S:
            _detected, _detected_at = detect_runs(), now
        for pid, info in _detected.items():
            if pid not in _runs:
                _runs[pid] = _Run(pid, info["config"], info["log"], info["started"])
        out = []
        live = [pid for pid, info in _detected.items() if _alive(pid, info["started"])]
        for pid, run in list(_runs.items()):
            alive = run.ended is None and _alive(pid, run.started)
            run.read()
            if not alive and run.ended is None:
                run.ended = time.time()
                _detected.pop(pid, None)
            # A new run replaces finished ones.
            if run.ended and (live or time.time() - run.ended > FINISHED_KEEP_S):
                del _runs[pid]
                continue
            out.append(run.summary(alive))
    return {"detected": bool(out), "runs": out}


_latest: dict = {"detected": False, "runs": []}
_sampler: threading.Thread | None = None


def get_training_info() -> dict:
    return _latest


def start_sampler(get_cfg) -> None:
    """Start the scrape loop once. get_cfg() returns the current [training] config."""
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
                log.exception("training probe failed")
                _latest = {"detected": False, "runs": [], "error": str(err)}
            time.sleep(cfg["scrape_interval_s"])

    _sampler = threading.Thread(target=loop, name="train-sampler", daemon=True)
    _sampler.start()
