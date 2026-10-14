import spark_dashboard.config as config


def _cfg(raw):
    return config._validate(config._merge(config.DEFAULTS, raw))


def test_defaults_are_one_local_node_with_every_card():
    cfg = _cfg({})
    assert cfg["nodes"] == [{"name": "", "url": "local"}]
    assert cfg["board"]["cards"] == list(config.CARDS)


def test_unknown_cards_and_metrics_are_dropped_in_order():
    cfg = _cfg({"board": {"cards": ["disk", "nope", "gpu"]}, "vllm": {"metrics": ["ttft", "x"]}})
    assert cfg["board"]["cards"] == ["disk", "gpu"]
    assert cfg["vllm"]["metrics"] == ["ttft"]


def test_string_where_list_expected_is_an_error_not_characters():
    import pytest
    with pytest.raises(TypeError):
        _cfg({"board": {"cards": "gpu"}})
    with pytest.raises(TypeError):
        _cfg({"nodes": {"url": "local"}})


def test_nodes_without_string_url_are_dropped_and_urls_normalised():
    cfg = _cfg({"nodes": [{"url": 8500}, {"name": "b", "url": "http://b:8500/"}]})
    assert cfg["nodes"] == [{"name": "b", "url": "http://b:8500"}]


def test_more_than_max_nodes_is_truncated():
    cfg = _cfg({"nodes": [{"url": f"http://n{i}"} for i in range(6)]})
    assert len(cfg["nodes"]) == config.MAX_NODES


def test_rate_window_spans_two_scrapes_and_refresh_has_a_floor():
    cfg = _cfg({"vllm": {"rate_window_s": 3, "scrape_interval_s": 5}, "board": {"refresh_ms": 10}})
    assert cfg["vllm"]["rate_window_s"] == 10
    assert cfg["board"]["refresh_ms"] == 500


def test_merge_is_deep_and_does_not_touch_defaults():
    cfg = _cfg({"cards": {"gpu": {"processes": 0}}})
    assert cfg["cards"]["gpu"]["processes"] == 0
    assert cfg["cards"]["cpu"]["per_core"] is True
    assert config.DEFAULTS["cards"]["gpu"]["processes"] == 5


def test_reload_on_change_and_keep_previous_on_bad_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[board]\ntitle = "ONE"\n')
    monkeypatch.setenv("SPARK_DASHBOARD_CONFIG", str(path))
    monkeypatch.setattr(config, "_cache", None)
    assert config.get_config()["board"]["title"] == "ONE"

    path.write_text('[board]\ntitle = "TWO"\n')
    import os
    os.utime(path, (1, 2))  # distinct mtime regardless of clock resolution
    assert config.get_config()["board"]["title"] == "TWO"

    path.write_text('[board]\ntitle = "BROKEN\n')
    os.utime(path, (1, 3))
    assert config.get_config()["board"]["title"] == "TWO"

    path.write_text('[board]\ncards = "gpu"\n')  # not a parse error: a type error
    os.utime(path, (1, 4))
    assert config.get_config()["board"]["title"] == "TWO"


def test_theme_defaults_to_classic_and_unknown_names_fall_back():
    assert _cfg({})["board"]["theme"] == "classic"
    assert _cfg({"board": {"theme": "wall"}})["board"]["theme"] == "wall"
    assert _cfg({"board": {"theme": "neon"}})["board"]["theme"] == "classic"


def test_theme_names_match_the_header_picker():
    from pathlib import Path
    import re
    html = (Path(config.__file__).parent / "static/index.html").read_text()
    css = (Path(config.__file__).parent / "static/style.css").read_text()
    assert re.findall(r'<option value="(\w+)"', html) == list(config.THEMES)
    for name in config.THEMES[1:]:
        assert f'[data-theme="{name}"]' in css
