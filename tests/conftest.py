"""Hardware-free test setup.

collectors imports SparkView's layers and the GPU wrappers at module level;
none of that exists off a GB10, so stub the modules before import. The
functions under test here never call them.
"""

import sys
import types

import pytest


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


try:
    import psutil  # noqa: F401
except ImportError:  # the Mac board venv has none; the tests never call it
    _stub("psutil")

_stub("sparkview")
_stub("sparkview.layers", power_rails=_stub("sparkview.layers.power_rails", read=lambda: None))
_stub("sparkview.layers.cpu", get_cpu_info=dict)
_stub("sparkview.layers.info", get_info=dict)
_stub("sparkview.layers.memory", get_memory=dict)
_stub("sparkview.layers.power", get_power_info=dict)
_stub("sparkview.layers.pressure", get_pressure=dict)
_stub("sparkview.layers.network", get_net_info=list)
_stub("sparkview.layers.gpu", NVITOP_AVAILABLE=False, is_coherent_uma=lambda d: True)
_stub("sparkview.layers.throttle", CLOCK_THRESHOLD_MHZ=1000, LOAD_GATE_PCT=10, query_nvidia_smi=list)


@pytest.fixture
def clock(monkeypatch):
    """Fake monotonic clock for the windowing code."""
    class Clock:
        t = 1000.0

        def now(self):
            return self.t

        def advance(self, s):
            self.t += s

    c = Clock()
    import spark_dashboard.collectors as collectors
    import spark_dashboard.vllm_probe as probe
    monkeypatch.setattr(collectors.time, "monotonic", c.now)
    monkeypatch.setattr(probe.time, "monotonic", c.now)
    return c
