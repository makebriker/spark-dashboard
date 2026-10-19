import math

import pytest

import spark_dashboard.vllm_probe as probe

EXPOSITION = """\
# HELP vllm:num_requests_running Number of requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="m"}} 2.0
vllm:num_requests_running{{engine="1",model_name="m"}} 1.0
vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} 0.2
vllm:kv_cache_usage_perc{{engine="1",model_name="m"}} 0.4
vllm:generation_tokens_total{{model_name="m"}} {gen}
vllm:prompt_tokens_total{{model_name="m"}} 100
vllm:time_to_first_token_seconds_bucket{{le="0.1",model_name="m"}} 3
vllm:time_to_first_token_seconds_sum{{model_name="m"}} {ttft_sum}
vllm:time_to_first_token_seconds_count{{model_name="m"}} {ttft_count}
vllm:time_to_first_token_seconds_created{{model_name="m"}} 1.7e9
vllm:inter_token_latency_seconds_sum{{model_name="m"}} {itl_sum}
vllm:inter_token_latency_seconds_count{{model_name="m"}} 1
vllm:e2e_request_latency_seconds_sum{{model_name="m"}} NaN
vllm:e2e_request_latency_seconds_count{{model_name="m"}} 0
"""


def exposition(gen=0, ttft_sum=0, ttft_count=0, itl_sum=0):
    return EXPOSITION.format(gen=gen, ttft_sum=ttft_sum, ttft_count=ttft_count, itl_sum=itl_sum)


def test_parse_sums_counters_means_level_gauges_skips_buckets_created_nan():
    values, labels = probe.parse_prometheus(exposition(gen=50))
    assert values["vllm:num_requests_running"] == 3.0          # summed across engines
    assert values["vllm:kv_cache_usage_perc"] == pytest.approx(0.3)  # averaged
    assert values["vllm:generation_tokens_total"] == 50
    assert not any("_bucket" in k or k.endswith("_created") for k in values)
    assert "vllm:e2e_request_latency_seconds_sum" not in values  # NaN dropped
    assert labels["model_name"] == "m"


def test_port_and_server_detection_from_cmdline():
    assert probe._is_vllm_server(["/usr/local/bin/vllm", "serve", "x", "--port", "8888"])
    assert probe._port_from_cmdline(["vllm", "serve", "x", "--port=8888"]) == 8888
    assert probe._port_from_cmdline(["vllm", "serve", "x"]) == probe.DEFAULT_PORT
    assert not probe._is_vllm_server(["vllm", "serve", "x", "--headless"])
    assert probe._is_vllm_server(["python", "-m", "vllm.entrypoints.openai.api_server"])


class Server:
    """Scripted /metrics responses: a text, or an exception to raise."""

    def __init__(self, monkeypatch):
        self.body = exposition()
        monkeypatch.setattr(probe, "_http_get", self.get)

    def get(self, url):
        if url.endswith("/v1/models"):
            raise OSError("401")
        if isinstance(self.body, Exception):
            raise self.body
        return self.body.encode()


def test_rates_over_the_window_and_counter_reset(clock, monkeypatch):
    srv = Server(monkeypatch)
    ep = probe._Endpoint("http://x")
    first = ep.collect(10)
    assert first["ok"] and first["metrics"]["decode_tps"] is None  # one sample: no rate yet

    for gen in (50, 100, 150):          # 10 tok/s, scraped every 5 s
        clock.advance(5)
        n = gen // 50                   # one request finished its first token per scrape
        srv.body = exposition(gen=gen, ttft_count=n, ttft_sum=0.5 * n, itl_sum=(gen - n) / 20)
        out = ep.collect(10)
    m = out["metrics"]
    assert out["window_s"] == 10        # oldest sample kept is exactly window_s back
    assert m["decode_tps"] == pytest.approx(10.0)
    assert m["running"] == 3 and m["kv_cache"] == pytest.approx(30)
    assert m["ttft"] == pytest.approx(0.5)
    # Per-stream speed: decode tokens (minus first tokens) over ITL seconds.
    assert m["stream_tps"] == pytest.approx(20.0)
    assert m["itl"] == pytest.approx(0.05)
    assert out["sample_at"]

    clock.advance(5)                    # server restarted: counters went backwards
    srv.body = exposition(gen=5)
    out = ep.collect(10)
    assert out["ok"] and out["metrics"]["decode_tps"] is None and out["window_s"] == 0


def test_timeout_keeps_last_values_long_but_refused_connection_does_not(clock, monkeypatch):
    srv = Server(monkeypatch)
    ep = probe._Endpoint("http://x")
    good = ep.collect(10)

    srv.body = TimeoutError("timed out")          # busy server: long grace
    clock.advance(probe.STALE_MAX_S + 10)
    out = ep.collect(10)
    assert out["ok"] and out["stale"] and out["metrics"] == good["metrics"]
    clock.advance(probe.STALE_TIMEOUT_MAX_S)
    assert ep.collect(10)["ok"] is False

    srv.body = exposition(gen=1)
    clock.advance(1)
    assert ep.collect(10)["ok"] and "stale" not in ep.collect(10)

    srv.body = ConnectionRefusedError("refused")  # no server: short grace
    clock.advance(probe.STALE_MAX_S - 1)
    assert ep.collect(10)["stale"]
    clock.advance(2)
    assert ep.collect(10)["ok"] is False


TF_CMD = (
    "python -m tensorfold serve /model --tp 2 --rank 0 --master 192.168.100.20 "
    "--master-port 29571 --context 262144 --parallel 4 --kv-dtype fp8 "
    "--no-update-check --host 127.0.0.1 --port 8000 --name DeepSeek-V4.1-Flash-TF"
).split()

TF_EXPOSITION = """\
# HELP tensorfold_requests_total Requests
# TYPE tensorfold_requests_total counter
tensorfold_requests_total{{model="DeepSeek-V4.1-Flash-TF"}} {reqs}
tensorfold_request_errors_total{{model="DeepSeek-V4.1-Flash-TF"}} 0
tensorfold_prompt_tokens_total{{model="DeepSeek-V4.1-Flash-TF"}} {prompt}
tensorfold_cached_tokens_total{{model="DeepSeek-V4.1-Flash-TF"}} {cached}
tensorfold_completion_tokens_total{{model="DeepSeek-V4.1-Flash-TF"}} {gen}
tensorfold_decode_rounds_total{{model="DeepSeek-V4.1-Flash-TF"}} {rounds}
tensorfold_decode_seconds_total{{model="DeepSeek-V4.1-Flash-TF"}} {dec}
tensorfold_prefill_seconds_total{{model="DeepSeek-V4.1-Flash-TF"}} 0
tensorfold_requests_inflight{{model="DeepSeek-V4.1-Flash-TF"}} 2
tensorfold_engine_fatal{{model="DeepSeek-V4.1-Flash-TF"}} 0
tensorfold_uptime_seconds{{model="DeepSeek-V4.1-Flash-TF"}} 908.6
"""


def tf_exposition(reqs=0, prompt=0, cached=0, gen=0, rounds=0, dec=0):
    return TF_EXPOSITION.format(reqs=reqs, prompt=prompt, cached=cached, gen=gen, rounds=rounds, dec=dec)


def test_tensorfold_detection_from_cmdline(monkeypatch):
    assert probe._is_tensorfold_server(TF_CMD)
    assert not probe._is_vllm_server(TF_CMD)
    assert probe._port_from_cmdline(TF_CMD) == 8000
    # Rank 1 runs on the other node and opens no API port.
    rank1 = [("1" if a == "0" and TF_CMD[i - 1] == "--rank" else a) for i, a in enumerate(TF_CMD)]
    assert "1" in rank1 and not probe._is_tensorfold_server(rank1)
    # Installed console script, with and without a full path.
    assert probe._is_tensorfold_server(["tensorfold", "serve", "/model", "--port", "8123"])
    assert probe._is_tensorfold_server(["/home/lesha/.venv/bin/tensorfold", "serve", "/model"])
    assert not probe._is_tensorfold_server(["tensorfold", "version"])
    assert not probe._is_tensorfold_server(["python", "-m", "http.server", "serve"])

    class Proc:
        def __init__(self, cmd):
            self.info = {"cmdline": cmd}

    monkeypatch.setattr(probe.psutil, "process_iter", lambda attrs: [Proc(TF_CMD), Proc(rank1)], raising=False)
    assert probe.detect_endpoints() == ["http://127.0.0.1:8000"]
    assert probe._cmdline_max_len == {"http://127.0.0.1:8000": 262144}


def test_vllm_detection_still_reads_max_model_len(monkeypatch):
    class Proc:
        info = {"cmdline": ["vllm", "serve", "x", "--port", "8888", "--max-model-len", "4096"]}

    monkeypatch.setattr(probe.psutil, "process_iter", lambda attrs: [Proc()], raising=False)
    assert probe.detect_endpoints() == ["http://127.0.0.1:8888"]
    assert probe._cmdline_max_len == {"http://127.0.0.1:8888": 4096}


def test_tensorfold_collect_rates_and_stream_speed(clock, monkeypatch):
    srv = Server(monkeypatch)
    srv.body = tf_exposition(reqs=17, prompt=1000, cached=0, gen=1000, rounds=4000, dec=25)
    ep = probe._Endpoint("http://x")
    assert ep.collect(10)["metrics"]["decode_tps"] is None

    clock.advance(10)
    srv.body = tf_exposition(reqs=19, prompt=1400, cached=100, gen=1500, rounds=4300, dec=27)
    out = ep.collect(10)
    m = out["metrics"]
    assert out["ok"] and out["model"] == "DeepSeek-V4.1-Flash-TF"
    assert m["decode_tps"] == pytest.approx(50.0)
    assert m["prefill_tps"] == pytest.approx(40.0)
    assert m["running"] == 2
    assert m["prefix_hit"] == pytest.approx(25.0)          # 100 cached / 400 prompt tokens
    assert m["req_rate"] == pytest.approx(0.2)
    assert m["stream_tps"] == pytest.approx(250.0)         # 500 tokens / 2 decode-seconds
    assert m["itl"] == pytest.approx(1 / 250)
    # No histograms or KV gauge on TensorFold: nothing is invented.
    assert m["waiting"] is None and m["kv_cache"] is None and m["ttft"] is None

    clock.advance(10)                                      # idle: decode clock did not move
    assert ep.collect(10)["metrics"]["stream_tps"] is None


def test_tensorfold_labels_parse():
    values, labels = probe.parse_prometheus(tf_exposition(gen=7))
    assert labels == {"model": "DeepSeek-V4.1-Flash-TF"}
    assert values["tensorfold_completion_tokens_total"] == 7
