"""init_df returns 3 values in deepfilternet 0.5.6 and 4 in the repo's main; _load takes either."""
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.denoise import dfn3  # noqa: E402


class _Model:
    def eval(self):
        return self

    def to(self, _device):
        return self


class _State:
    def sr(self):
        return 48000


@pytest.mark.parametrize("returned", [
    (_Model(), _State(), "suffix"),
    (_Model(), _State(), "suffix", 120),
])
def test_load_accepts_three_or_four_values_from_init_df(monkeypatch, returned):
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _s: None
    fake_df = types.ModuleType("df")
    fake_cfg = types.ModuleType("df.config")
    fake_cfg.config = types.SimpleNamespace(set=lambda *a, **k: None)
    fake_enh = types.ModuleType("df.enhance")
    fake_enh.init_df = lambda *a, **k: returned
    for name, mod in (("torch", fake_torch), ("df", fake_df), ("df.config", fake_cfg),
                      ("df.enhance", fake_enh)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(dfn3.DeepFilterNet3Backend, "check_available",
                        classmethod(lambda cls, cfg=None: (True, "")))
    backend = dfn3.DeepFilterNet3Backend({})
    backend._load()
    assert backend._df_state is returned[1] and backend._model is returned[0]
