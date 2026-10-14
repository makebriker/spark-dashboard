import pytest

import spark_dashboard.collectors as collectors


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(collectors, "_net_hist", {})
    monkeypatch.setattr(collectors, "_net_last_t", 0.0)


def feed(clock, dt, rates):
    """One cycle: every iface gets an instantaneous rate; returns the entries."""
    clock.advance(dt)
    entries = [{"iface": i, "rx_rate": r, "tx_rate": r * 2} for i, r in rates.items()]
    collectors.average_net_rates(entries)
    return {e["iface"]: e for e in entries}


def test_first_cycle_has_no_interval_and_keeps_instantaneous_values(clock):
    e = feed(clock, 0, {"eth0": 100.0})["eth0"]
    assert e["rx_rate"] == 100.0 and e["rx_rate_now"] == 100.0
    assert e["rate_window_s"] == 0


def test_window_is_exactly_15s_once_warm_and_time_weighted(clock):
    W = collectors.NET_AVG_WINDOW_S
    feed(clock, 0, {"eth0": 0.0})
    for _ in range(10):                      # 20 s of 100 B/s at 2 s samples
        e = feed(clock, 2, {"eth0": 100.0})["eth0"]
    assert e["rate_window_s"] == W
    assert e["rx_rate"] == pytest.approx(100.0)

    # A single 2 s burst of 1000 B/s inside a 15 s window: it contributes
    # exactly 2/15 of its rate, and is counted once, not again later.
    e = feed(clock, 2, {"eth0": 1000.0})["eth0"]
    assert e["rx_rate"] == pytest.approx((1000 * 2 + 100 * 13) / W)
    assert e["rx_rate_now"] == 1000.0
    for _ in range(7):
        e = feed(clock, 2, {"eth0": 100.0})["eth0"]
    assert e["rx_rate"] == pytest.approx((1000 * 1 + 100 * 14) / W)  # 1 s of it left
    e = feed(clock, 2, {"eth0": 100.0})["eth0"]
    assert e["rx_rate"] == pytest.approx(100.0)  # the burst has left the window


def test_a_long_stall_counts_only_the_part_inside_the_window(clock):
    feed(clock, 0, {"eth0": 0.0})
    e = feed(clock, 60, {"eth0": 50.0})["eth0"]  # one sample covering 60 s
    assert e["rate_window_s"] == collectors.NET_AVG_WINDOW_S
    assert e["rx_rate"] == pytest.approx(50.0) and e["tx_rate"] == pytest.approx(100.0)


def test_history_of_an_interface_that_disappears_is_dropped(clock):
    feed(clock, 0, {"eth0": 1.0, "wl0": 1.0})
    feed(clock, 2, {"eth0": 1.0, "wl0": 1.0})
    feed(clock, 2, {"eth0": 1.0})
    assert set(collectors._net_hist) == {"eth0"}


def test_safe_call_returns_default_and_logs_once(caplog):
    collectors._reported.clear()

    def boom():
        raise RuntimeError("nvml gone")

    with caplog.at_level("WARNING"):
        assert collectors._safe_call(boom, default=[]) == []
        assert collectors._safe_call(boom, default=[]) == []
    assert sum("nvml gone" in r.message for r in caplog.records) == 1
