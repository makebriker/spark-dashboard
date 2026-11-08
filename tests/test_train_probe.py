import json
import time

import pytest

import spark_dashboard.train_probe as probe

CFG = {"type": "sd_trainer", "training_folder": "/nonexistent", "save": {"save_every": 250},
       "train": {"steps": 1000}, "sample": {"sample_every": 250, "prompts": ["a"]}}


def bar(step, elapsed, loss=None, name="run_v1", total=1000):
    post = f", 15.00s/it, lr: 1.0e-04 loss: {loss:.3e}" if loss is not None else ", ?it/s"
    return f"{name}:   7%|7         | {step}/{total} [{elapsed}<4:04:54{post}]"


def header():
    return "Running 1 job\n" + json.dumps(CFG, indent=4) + "\n"


def train(steps, start_s=0, every=15, first=1):
    """The two lines tqdm writes per step: postfix refresh, then the increment."""
    out = []
    t = start_s
    for n in range(first, first + steps):
        t += every
        out.append(bar(n - 1, f"{t // 60:02d}:{t % 60:02d}", loss=0.04))
        out.append(bar(n, f"{t // 60:02d}:{t % 60:02d}", loss=0.04))
    return out, t


def bars(lines):
    """As tqdm writes them: "\r" before each bar, none after the newest."""
    return "".join("\r" + line for line in lines)


@pytest.fixture(autouse=True)
def fresh_module(monkeypatch):
    monkeypatch.setattr(probe, "_runs", {})
    monkeypatch.setattr(probe, "_detected", {})
    monkeypatch.setattr(probe, "_workers", set())
    monkeypatch.setattr(probe, "_detected_at", -probe.DETECT_TTL_S)


def write_run(tmp_path, text, name="train.log"):
    path = tmp_path / name
    path.write_bytes(text.encode())
    run = probe._Run(42, "config/run_v1.yaml", str(path), time.time() - 100)
    run.read()
    return run, path


def test_reads_config_name_steps_and_rate(tmp_path):
    lines, _ = train(30)
    run, _ = write_run(tmp_path, header() + bars(lines))
    s = run.summary(True)
    assert run.cfg["save"]["save_every"] == 250
    assert s["name"] == "run_v1" and s["state"] == "TRAINING"
    assert (s["step"], s["total"]) == (30, 1000)
    assert s["s_per_step"] == 15.0
    assert s["next_sample"] == 250 and s["next_save"] == 250
    assert len(s["loss_points"]) == 30 and s["loss"] == pytest.approx(0.04)


def test_sample_round_is_left_out_of_the_rate_and_counted_in_the_eta(tmp_path):
    a, t = train(10)
    samples = ["Generating Samples:   0%|          | 0/3 [00:00<?, ?it/s]",
               "Generating Samples: 100%|##########| 3/3 [07:30<00:00, 150.00s/it]"]
    b, _ = train(10, start_s=t + 450, first=11)
    run, _ = write_run(tmp_path, header() + bars(a + samples + b))
    s = run.summary(True)
    assert s["s_per_step"] == 15.0
    assert s["sample_s"] == 450
    # 980 steps at 15 s, plus samples at 250, 500, 750 and after the last step.
    assert s["eta_s"] == 980 * 15 + 4 * 450


def test_log_is_read_incrementally_with_a_split_line(tmp_path):
    lines, _ = train(5)
    text = header() + bars(lines)
    cut = len(text) - 20
    run, path = write_run(tmp_path, text[:cut])
    assert run.step == 4
    path.write_bytes(text.encode())
    run.read()
    assert run.step == 5


@pytest.mark.parametrize("line, phase", [
    ("Caching latents to disk:  10%|#         | 14/140 [01:40<02:26,  1.16s/it]", "CACHING"),
    ("Loading weights:  50%|#####     | 1/2 [00:10<00:10, 10.00s/it]", "LOADING"),
    ("Generating Samples:  33%|###3      | 1/3 [02:27<04:55, 147.57s/it]", "SAMPLING"),
])
def test_other_bars_set_the_phase(tmp_path, line, phase):
    run, _ = write_run(tmp_path, header() + line + "\r")
    s = run.summary(True)
    assert s["state"] == phase and s["phase_bar"]["label"] == line.split(":")[0]


def test_saving_and_checkpoints(tmp_path):
    lines, _ = train(3)
    text = header() + bars(lines) + "\nSaving at step 250\nSaved checkpoint to /x/run_v1_000000250.safetensors\n"
    run, _ = write_run(tmp_path, text)
    s = run.summary(True)
    assert s["state"] == "SAVING"
    assert s["checkpoint"] == {"step": 250, "final": False, "ago_s": None}


def test_ended_runs_are_done_stopped_or_failed(tmp_path):
    lines, _ = train(3, first=998)
    done, _ = write_run(tmp_path, header() + bars(lines) +
                        "\nSaved checkpoint to /x/run_v1.safetensors\n", "done.log")
    assert done.summary(False)["state"] == "DONE"
    assert done.summary(False)["checkpoint"]["final"]

    lines, _ = train(3)
    stopped, _ = write_run(tmp_path, header() + bars(lines), "stopped.log")
    assert stopped.summary(False)["state"] == "STOPPED"

    failed, _ = write_run(tmp_path, header() + bars(lines) +
                          "\nTraceback (most recent call last):\n  File \"run.py\"\ntorch.OutOfMemoryError: CUDA out of memory\n",
                          "failed.log")
    s = failed.summary(False)
    assert s["state"] == "FAILED" and s["error"] == "torch.OutOfMemoryError: CUDA out of memory"
    assert s["eta_s"] is None and s["next_sample"] is None


def test_loss_curve_is_capped(tmp_path):
    lines, _ = train(400)
    run, _ = write_run(tmp_path, header() + bars(lines))
    assert len(run.summary(True)["loss_points"]) == probe.LOSS_POINTS


def test_no_log_file_still_reports_the_run():
    run = probe._Run(42, "config/run_v1.yaml", None, time.time() - 100)
    run.read()
    s = run.summary(True)
    assert s["name"] == "run_v1" and s["log_error"] == "stdout is not a file"


def test_truncated_log_starts_over(tmp_path):
    lines, _ = train(20)
    run, path = write_run(tmp_path, header() + bars(lines))
    lines, _ = train(2)
    path.write_bytes((header() + bars(lines)).encode())
    run.read()
    assert run.step == 2


@pytest.mark.parametrize("cmd, cwd, ok", [
    (["/home/l/ai-toolkit/.venv/bin/python", "run.py", "config/a.yaml"], "toolkit", True),
    ([".venv/bin/python", "run.py", "--log", "config/a.yaml"], "toolkit", True),
    (["python3", "run.py", "config/a.yaml"], "plain", False),
    (["python3", "run.py"], "toolkit", False),
    (["python3", "serve.py", "config/a.yaml"], "toolkit", False),
    (["bash", "run.py", "config/a.yaml"], "toolkit", False),
])
def test_detects_ai_toolkit_by_its_checkout(tmp_path, cmd, cwd, ok):
    (tmp_path / "toolkit" / "toolkit").mkdir(parents=True)
    (tmp_path / "plain").mkdir()
    assert bool(probe._is_toolkit_run(cmd, str(tmp_path / cwd))) is ok


def test_disabled_clears_runs(monkeypatch):
    monkeypatch.setattr(probe, "_runs", {1: object()})
    monkeypatch.setattr(probe, "detect_runs", lambda: pytest.fail("scanned"))
    assert probe._collect_all({"enabled": False}) == {"detected": False, "runs": []}
    assert probe._runs == {}


def test_finished_run_stays_until_a_new_one(tmp_path, monkeypatch):
    lines, _ = train(3)
    (tmp_path / "a.log").write_text(header() + bars(lines))
    (tmp_path / "b.log").write_text(header() + bars(lines))
    alive = {1}
    found = {1: {"config": "config/a.yaml", "log": str(tmp_path / "a.log"), "started": time.time()}}
    monkeypatch.setattr(probe, "DETECT_TTL_S", 0)
    monkeypatch.setattr(probe, "detect_runs", lambda: {p: found[p] for p in alive if p in found})
    monkeypatch.setattr(probe, "_alive", lambda pid, started=None: pid in alive)

    assert [r["state"] for r in probe._collect_all({"enabled": True})["runs"]] == ["TRAINING"]
    alive.clear()
    assert [r["state"] for r in probe._collect_all({"enabled": True})["runs"]] == ["STOPPED"]
    found[2] = {"config": "config/b.yaml", "log": str(tmp_path / "b.log"), "started": time.time()}
    alive.add(2)
    runs = probe._collect_all({"enabled": True})["runs"]
    assert [(r["pid"], r["state"]) for r in runs] == [(2, "TRAINING")]


def test_newest_bar_counts_before_the_next_write(tmp_path):
    run, _ = write_run(tmp_path, header() + bars(train(3)[0] + [
        "Generating Samples:   0%|          | 0/1 [00:00<?, ?it/s]"]))
    assert run.summary(True)["state"] == "SAMPLING"


def test_print_output_sharing_a_line_with_a_bar(tmp_path):
    lines, _ = train(3)
    text = header() + bars(lines) + "Saving at step 250\nSaved checkpoint to /x/run_v1_000000250.safetensors\n"
    run, _ = write_run(tmp_path, text)
    assert run.summary(True)["state"] == "SAVING"
    assert run.summary(True)["checkpoint"]["step"] == 250


def test_traceback_before_more_training_is_not_fatal(tmp_path):
    a, t = train(3)
    b, _ = train(3, first=998, start_s=t)
    text = header() + bars(a) + "\nTraceback (most recent call last):\nWarning: x\n" + bars(b) + \
        "\nSaved checkpoint to /x/run_v1.safetensors\n"
    run, _ = write_run(tmp_path, text)
    assert run.summary(False)["state"] == "DONE"


def test_ctrl_c_reads_as_stopped(tmp_path):
    lines, _ = train(3)
    run, _ = write_run(tmp_path, header() + bars(lines) +
                       "\nTraceback (most recent call last):\n  File \"run.py\"\nKeyboardInterrupt\n")
    assert run.summary(False)["state"] == "STOPPED"


def test_appended_log_starts_over_at_the_new_job(tmp_path):
    old, _ = train(50)
    new, _ = train(3)
    text = header() + bars(old) + "\nTraceback (most recent call last):\nRuntimeError: oom\n" + header() + bars(new)
    run, _ = write_run(tmp_path, text)
    s = run.summary(True)
    assert s["step"] == 3 and len(run.losses) == 3 and not run.traceback


def test_utf8_split_across_reads(tmp_path):
    name = "run_é"
    lines = [bar(1, "00:15", 0.04, name=name), bar(2, "00:30", 0.04, name=name)]
    data = (header() + bars(lines)).encode()
    cut = data.index("é".encode()) + 1          # inside the two-byte character
    path = tmp_path / "train.log"
    path.write_bytes(data[:cut])
    run = probe._Run(42, "config/run_v1.yaml", str(path), time.time())
    run.read()
    path.write_bytes(data)
    run.read()
    assert run.name == name and run.step == 2


def test_truncation_keeps_the_end_time(tmp_path):
    lines, _ = train(20)
    run, path = write_run(tmp_path, header() + bars(lines))
    run.ended = 123.0
    path.write_text(header())
    run.read()
    assert run.ended == 123.0 and run.step == 0


class FakeProc:
    def __init__(self, pid, ppid, cmd, cwd):
        self.info = {"pid": pid, "ppid": ppid, "cmdline": cmd, "create_time": 1000.0 + pid}
        self._cwd = cwd

    def cwd(self):
        return self._cwd


def test_workers_are_skipped_even_after_reparenting(tmp_path, monkeypatch):
    (tmp_path / "toolkit").mkdir()
    cmd = [".venv/bin/python", "run.py", "config/a.yaml"]
    procs = [FakeProc(10, 1, cmd, str(tmp_path)), FakeProc(11, 10, cmd, str(tmp_path)),
             FakeProc(12, 1, ["python3", "other.py"], str(tmp_path))]
    monkeypatch.setattr(probe.psutil, "process_iter", lambda attrs: procs)
    monkeypatch.setattr(probe.os, "readlink", lambda path: (_ for _ in ()).throw(OSError()))
    assert list(probe.detect_runs()) == [10]
    procs[:] = [FakeProc(11, 1, cmd, str(tmp_path))]   # the run exited, its worker lives on
    assert probe.detect_runs() == {}
