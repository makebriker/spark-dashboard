import json
import threading
import time

import pytest

import spark_dashboard.comfy_probe as probe

GRAPH = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux/krea2_turbo.safetensors"}},
    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "style.safetensors"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
}


def job(prompt_id, graph=None):
    return [1, prompt_id, graph or GRAPH, {"create_time": 1794074449000}, ["9"]]


def history(prompt_id="done", events=None, status_str="success"):
    events = events if events is not None else [
        ["execution_start", {"timestamp": 1794074449000}],
        ["execution_success", {"timestamp": 1794074491500}],
    ]
    return {prompt_id: {
        "prompt": job(prompt_id),
        "outputs": {},
        "status": {"status_str": status_str, "completed": True, "messages": events},
        "meta": {},
    }}


class Server:
    """Scripted ComfyUI: /queue and /history bodies, or an exception to raise."""

    def __init__(self, monkeypatch):
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = history()
        self.history_reads = 0
        monkeypatch.setattr(probe, "_http_get", self.get)
        monkeypatch.setattr(probe._Progress, "start", lambda _self: None)

    def get(self, url):
        if url.endswith("/queue"):
            body = self.queue
        else:
            self.history_reads += 1
            body = self.history
        if isinstance(body, Exception):
            raise body
        return json.dumps(body).encode()


@pytest.fixture
def server(monkeypatch):
    return Server(monkeypatch)


@pytest.mark.parametrize("cmd, cwd", [
    (["/home/lesha/comfyui-env/bin/python", "main.py", "--listen", "0.0.0.0"], None),
    (["/opt/py/bin/python", "/home/lesha/ComfyUI/main.py"], None),
    (["python3", "main.py"], "/srv/ComfyUI"),
])
def test_detects_comfyui_from_interpreter_script_or_cwd(cmd, cwd):
    assert probe._is_comfy_server(cmd, cwd)


@pytest.mark.parametrize("cmd, cwd", [
    (["python3", "main.py"], "/srv/other"),
    (["/home/lesha/comfyui-env/bin/python", "train.py"], None),
    (["vllm", "serve", "x"], "/home/lesha/ComfyUI"),
    (["vim", "main.py"], "/home/lesha/ComfyUI"),
    (["tail", "-f", "/home/lesha/ComfyUI/main.py"], None),
    (["python3", "tools/check.py", "main.py"], "/home/lesha/ComfyUI"),
])
def test_other_processes_are_not_comfyui(cmd, cwd):
    assert not probe._is_comfy_server(cmd, cwd)


def test_port_comes_from_cmdline_or_defaults():
    assert probe._port_from_cmdline(["python", "main.py", "--port", "8189"]) == 8189
    assert probe._port_from_cmdline(["python", "main.py", "--port=8189"]) == 8189
    assert probe._port_from_cmdline(["python", "main.py"]) == probe.DEFAULT_PORT


def test_model_is_the_main_loader_without_folder_or_extension():
    assert probe.job_model(GRAPH) == "krea2_turbo"
    assert probe.job_model({"1": {"inputs": {"ckpt_name": "a\\sd.ckpt"}}}) == "sd"


def test_model_counts_extra_loaders_and_ignores_bad_graphs():
    graph = {"1": {"inputs": {"unet_name": "a.gguf"}}, "2": {"inputs": {"unet_name": "b.gguf"}}}
    assert probe.job_model(graph) == "a +1"
    assert probe.job_model({"1": {"inputs": {"vae_name": "v.pt"}}}) is None
    assert probe.job_model(None) is None


def test_last_job_success_has_duration_and_model():
    last = probe.parse_last_job(history())
    assert (last["result"], last["duration_s"], last["model"]) == ("success", 42.5, "krea2_turbo")
    assert last["error"] is None


def test_last_job_interrupted_is_not_reported_as_error():
    last = probe.parse_last_job(history(status_str="error", events=[
        ["execution_start", {"timestamp": 1000}],
        ["execution_interrupted", {"timestamp": 3000, "node_type": "KSampler"}],
    ]))
    assert (last["result"], last["duration_s"], last["error"]) == ("interrupted", 2.0, None)


def test_last_job_error_keeps_the_first_line_and_its_node():
    last = probe.parse_last_job(history(status_str="error", events=[
        ["execution_error", {"timestamp": 3000, "node_type": "UNETLoader", "exception_message": "no file\ntrace"}],
    ]))
    assert (last["result"], last["error"], last["duration_s"]) == ("error", "UNETLoader: no file", None)


def test_no_history_means_no_last_job():
    assert probe.parse_last_job({}) is None


def test_collect_reports_queue_and_elapsed_from_first_sighting(server, clock):
    ep = probe._Endpoint("http://x")
    server.queue = {"queue_running": [job("a")], "queue_pending": [job("b"), job("c")]}
    first = ep.collect()
    assert (first["running"], first["pending"], first["model"]) == (1, 2, "krea2_turbo")
    assert first["job_elapsed_s"] == 0
    assert "stale" not in first
    clock.advance(12)
    assert ep.collect()["job_elapsed_s"] == 12


def test_collect_drops_graphs_from_its_payload(server, clock):
    server.queue = {"queue_running": [job("a")], "queue_pending": []}
    assert "class_type" not in json.dumps(probe._Endpoint("http://x").collect())


def test_new_job_restarts_elapsed_and_rereads_history(server, clock):
    ep = probe._Endpoint("http://x")
    server.queue = {"queue_running": [job("a")], "queue_pending": []}
    ep.collect()
    assert server.history_reads == 1
    clock.advance(5)
    ep.collect()
    assert server.history_reads == 1
    server.queue = {"queue_running": [job("b")], "queue_pending": []}
    assert ep.collect()["job_elapsed_s"] == 0
    assert server.history_reads == 2


def test_idle_server_rereads_history_on_its_ttl(server, clock):
    ep = probe._Endpoint("http://x")
    ep.collect()
    clock.advance(probe.HISTORY_TTL_S)
    ep.collect()
    assert server.history_reads == 2


def test_failed_history_keeps_the_queue_and_the_last_job(server, clock):
    ep = probe._Endpoint("http://x")
    ep.collect()
    clock.advance(probe.HISTORY_TTL_S)
    server.history = OSError("boom")
    out = ep.collect()
    assert out["ok"] and out["last_job"]["result"] == "success"


def test_failed_history_is_retried_on_the_next_scrape(server, clock):
    ep = probe._Endpoint("http://x")
    server.queue = {"queue_running": [job("a")], "queue_pending": []}
    ep.collect()
    server.queue = {"queue_running": [], "queue_pending": []}
    server.history = OSError("boom")
    ep.collect()
    clock.advance(3)
    server.history = history("a", status_str="error", events=[])
    assert ep.collect()["last_job"]["result"] == "error"


def test_refused_queue_stays_stale_then_goes_down(server, clock):
    ep = probe._Endpoint("http://x")
    ep.collect()
    server.queue = OSError("refused")
    clock.advance(probe.STALE_MAX_S - 1)
    stale = ep.collect()
    assert stale["ok"] and stale["stale"]
    clock.advance(2)
    assert not ep.collect()["ok"]


def test_timeout_gets_the_longer_grace(server, clock):
    ep = probe._Endpoint("http://x")
    ep.collect()
    server.queue = TimeoutError("busy")
    clock.advance(probe.STALE_TIMEOUT_MAX_S - 1)
    assert ep.collect()["stale"]


def test_answer_that_is_not_a_queue_counts_as_failed(server, clock):
    server.queue = {"detail": "not found"}
    assert not probe._Endpoint("http://x").collect()["ok"]


def step(prompt_id="a", node="4", value=12, total=30):
    return {"type": "progress", "data": {"value": value, "max": total, "prompt_id": prompt_id, "node": node}}


def test_progress_names_the_node_from_the_graph():
    progress = probe._Progress("http://x:8188")
    assert progress.ws_url == "ws://x:8188/ws"
    progress.feed(step())
    assert progress.for_job("a", GRAPH) == {"value": 12, "max": 30, "node_type": "KSampler"}


def test_progress_of_another_job_is_ignored():
    progress = probe._Progress("http://x")
    progress.feed(step(prompt_id="old"))
    assert progress.for_job("a", GRAPH) is None


def node_state(node, state):
    return {"type": "progress_state", "data": {"prompt_id": "a", "nodes": {node: {"state": state}}}}


def test_step_stays_while_its_node_runs_and_ignores_other_nodes():
    progress = probe._Progress("http://x")
    progress.feed(step())
    progress.feed(node_state("4", "running"))
    progress.feed(node_state("9", "finished"))
    assert progress.for_job("a", GRAPH)


@pytest.mark.parametrize("state", ["finished", "error"])
def test_step_clears_when_its_node_ends(state):
    progress = probe._Progress("http://x")
    progress.feed(step())
    progress.feed(node_state("4", state))
    assert progress.latest is None


def test_monitor_frames_and_bad_steps_are_ignored():
    progress = probe._Progress("http://x")
    progress.feed({"type": "crystools.monitor", "data": {"cpu_utilization": 1}})
    progress.feed({"type": "status"})
    assert progress.latest is None
    progress.feed(step(total=0))
    assert progress.for_job("a", GRAPH) is None


def test_node_missing_from_the_graph_has_no_type():
    progress = probe._Progress("http://x")
    progress.feed(step(node="12:3"))
    assert progress.for_job("a", GRAPH)["node_type"] is None


def test_collect_adds_the_step_of_the_running_job(server, clock):
    ep = probe._Endpoint("http://x")
    server.queue = {"queue_running": [job("a")], "queue_pending": []}
    ep.progress.feed(step())
    out = ep.collect()
    assert out["progress"]["value"] == 12
    assert out["progress_private"] is False


def test_job_with_a_client_id_is_marked_private(server, clock):
    item = job("a")
    item[3]["client_id"] = "browser"
    server.queue = {"queue_running": [item], "queue_pending": []}
    out = probe._Endpoint("http://x").collect()
    assert out["progress"] is None and out["progress_private"] is True


def test_null_client_id_is_not_private(server, clock):
    # ComfyUI broadcasts progress when client_id is null.
    item = job("a")
    item[3]["client_id"] = None
    server.queue = {"queue_running": [item], "queue_pending": []}
    assert probe._Endpoint("http://x").collect()["progress_private"] is False


@pytest.mark.parametrize("value, total", [(1, float("nan")), (float("inf"), 30), (1, float("inf")), (True, 30)])
def test_non_finite_steps_are_dropped(value, total):
    progress = probe._Progress("http://x")
    progress.feed(step(value=value, total=total))
    assert progress.for_job("a", GRAPH) is None


def test_stale_answer_keeps_the_step_moving(server, clock):
    ep = probe._Endpoint("http://x")
    server.queue = {"queue_running": [job("a")], "queue_pending": []}
    ep.progress.feed(step(value=3))
    ep.collect()
    server.queue = TimeoutError("busy")
    ep.progress.feed(step(value=9))
    out = ep.collect()
    assert out["stale"] and out["progress"]["value"] == 9


def test_disabling_closes_every_reader(server, monkeypatch):
    ep = probe._Endpoint("http://x")
    monkeypatch.setattr(probe, "_endpoints", {"http://x": ep})
    probe._collect_all({"enabled": False, "endpoints": []})
    assert probe._endpoints == {}
    assert ep.progress.stop.is_set()


def test_removed_endpoint_stops_its_reader(server, monkeypatch):
    ep = probe._Endpoint("http://x")
    monkeypatch.setattr(probe, "_endpoints", {"http://x": ep})
    probe._collect_all({"enabled": True, "endpoints": ["http://y"]})
    assert "http://x" not in probe._endpoints
    assert ep.progress.stop.is_set()


def test_reader_takes_steps_skips_binary_and_clears_on_disconnect():
    server_mod = pytest.importorskip("websockets.sync.server")
    seen = threading.Event()

    def handler(ws):
        ws.send(b"\x00\x00\x00\x01preview")
        ws.send(json.dumps({"type": "crystools.monitor", "data": {}}))
        ws.send(json.dumps(step()))
        seen.wait(5)

    with server_mod.serve(handler, "127.0.0.1", 0) as ws_server:
        port = ws_server.socket.getsockname()[1]
        threading.Thread(target=ws_server.serve_forever, daemon=True).start()
        progress = probe._Progress(f"http://127.0.0.1:{port}")
        progress.start()
        try:
            deadline = time.monotonic() + 5
            while progress.latest is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert progress.for_job("a", GRAPH)["value"] == 12
            seen.set()
            deadline = time.monotonic() + 5
            while progress.latest is not None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert progress.latest is None
        finally:
            progress.close()
            ws_server.shutdown()
        progress.thread.join(5)
        assert not progress.thread.is_alive()


def test_disabled_never_probes(monkeypatch):
    monkeypatch.setattr(probe, "detect_endpoints", lambda: pytest.fail("probed"))
    assert probe._collect_all({"enabled": False, "endpoints": []}) == {"detected": False, "servers": []}


def test_explicit_endpoints_skip_detection(server, monkeypatch):
    monkeypatch.setattr(probe, "_endpoints", {})
    monkeypatch.setattr(probe, "detect_endpoints", lambda: pytest.fail("scanned"))
    out = probe._collect_all({"enabled": True, "endpoints": ["http://x/"]})
    assert out["detected"]
    assert out["servers"][0]["endpoint"] == "http://x"
