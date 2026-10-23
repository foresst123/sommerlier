"""utils/acoustic_boundary.py is a verbatim copy from origin/solid-architecture."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.acoustic_boundary import AcousticBoundaryFinder  # noqa: E402

SR = 16000


def test_energy_fallback_cuts_in_the_pause():
    rng = np.random.default_rng(0)
    x = np.concatenate([0.3 * rng.standard_normal(SR), 1e-4 * rng.standard_normal(SR // 10),
                        0.3 * rng.standard_normal(SR)]).astype("float32")
    d = AcousticBoundaryFinder(x, SR).find_cut(int(0.98 * SR), search_min=int(0.9 * SR), search_max=int(1.2 * SR))
    assert d.method == "energy_pause" and SR <= d.sample <= SR + SR // 10 and not d.fallback


def test_no_pause_falls_back_to_the_requested_point():
    x = np.ones(SR, "float32")
    d = AcousticBoundaryFinder(x, SR).find_cut(SR // 2, search_min=SR // 2 - 100, search_max=SR // 2 + 100)
    assert d.fallback and d.sample == SR // 2
