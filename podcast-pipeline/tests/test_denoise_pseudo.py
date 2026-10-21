"""Pseudo-steps (blendNN) in denoise chains."""

import numpy as np
import pytest

from schemas.audio import AudioData
from services import denoise_service
from services.denoise_service import DenoiseService, chain_id_of
from models.denoise import pseudo
from models.denoise.pseudo import (backend_tokens, blend, describe, is_pseudo,
                                   parse_pseudo, validate_chain_tokens)
from tests.test_denoise_stage import FakeBackend, audio, speech, SR


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    """Real parse_chain/chain_id; fake backends 'half', 'plus1', 'nan', 'silent', 'boom'."""
    from models.denoise import chain_id, parse_chain
    FakeBackend.calls = []

    def get_backend(name, cfg=None, logger=None):
        if name not in ("half", "plus1", "nan", "silent", "boom", "identity"):
            raise KeyError(name)
        return FakeBackend(name, name)

    monkeypatch.setattr(denoise_service, "_load_api",
                        lambda: (get_backend, parse_chain, chain_id))


def run(spec, wav=None):
    data = audio(wav)
    before = data.waveform.copy()
    info = DenoiseService().apply(data, spec, None, None)
    return before, data.waveform, info


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("tok,val", [("blend5", 5), ("blend050", 50), ("blend0", 0),
                                     ("blend100", 100), ("blend25", 25), ("BLEND50", 50)])
def test_parse_valid(tok, val):
    assert parse_pseudo(tok) == ("blend", val) and is_pseudo(tok)


@pytest.mark.parametrize("tok", ["blend101", "blend-1", "blend", "blend1000", "blendx",
                                 "dfn3", "xblend50", "blend5 0"])
def test_parse_invalid(tok):
    assert parse_pseudo(tok) is None and not is_pseudo(tok)


def test_chain_parser_normalises_case_and_rejects_negative():
    from models.denoise import parse_chain
    assert parse_chain("DFN3>Blend50") == ["dfn3", "blend50"]
    with pytest.raises(ValueError):
        parse_chain("dfn3>blend-1")


def test_validate_tokens():
    validate_chain_tokens(["a", "blend50", "b"])
    validate_chain_tokens(["a", "blend100"])
    validate_chain_tokens([])
    with pytest.raises(ValueError, match="cannot be first"):
        validate_chain_tokens(["blend50", "a"])
    with pytest.raises(ValueError):
        validate_chain_tokens(["blend50"])
    assert backend_tokens(["a", "blend50", "b", "blend5"]) == ["a", "b"]
    assert "50%" in describe("blend50") and "dfn3" in describe("dfn3")


# --- blend math ------------------------------------------------------------

def test_blend_math():
    cur = np.linspace(-1, 1, 101, dtype=np.float32)
    raw = np.linspace(1, -1, 101, dtype=np.float32) * 0.5
    assert np.array_equal(blend(cur, raw, 0), cur)
    assert np.array_equal(blend(cur, raw, 100), raw)
    mid = blend(cur, raw, 50)
    assert mid.dtype == np.float32 and len(mid) == len(cur)
    assert np.allclose(mid, 0.5 * cur + 0.5 * raw, atol=1e-7)
    assert blend(cur.astype(np.float64), raw, 30).dtype == np.float32
    with pytest.raises(ValueError):
        blend(cur, raw[:-1], 50)


# --- the service -----------------------------------------------------------

def test_blend0_equals_chain_output_and_blend100_equals_input():
    before, out0, _ = run("half>blend0")
    assert np.array_equal(out0, (before * 0.5).astype(np.float32))
    before, out100, info = run("half>blend100")
    assert np.array_equal(out100, before) and info["errors"] == []


def test_blend50_midpoint_and_dtype():
    before, out, info = run("half>blend50")
    assert out.dtype == np.float32
    assert np.allclose(out, 0.75 * before, atol=1e-6)
    assert info["chain"] == ["half", "blend50"] and info["chain_id"] == "half+blend50"
    assert info["blend"] == {"percent": 50}


def test_blend_does_not_reach_get_backend(monkeypatch):
    seen = []
    orig = denoise_service._load_api()

    def spy(name, cfg=None, logger=None):
        seen.append(name)
        return orig[0](name, cfg, logger)

    monkeypatch.setattr(denoise_service, "_load_api", lambda: (spy, orig[1], orig[2]))
    run("half>plus1>blend30")
    assert seen == ["half", "plus1"] and FakeBackend.calls == ["half", "plus1"]


def test_ordering_and_steps_record():
    before, out, info = run("half>plus1>blend30")
    chain_out = (before * 0.5 + 0.01).astype(np.float32)
    assert np.allclose(out, 0.7 * chain_out + 0.3 * before, atol=1e-6)
    assert [(s["token"], s["kind"]) for s in info["steps"]] == [
        ("half", "backend"), ("plus1", "backend"), ("blend30", "blend")]
    assert all(s["seconds"] >= 0 for s in info["steps"])


def test_blend_between_steps_uses_the_chain_input():
    # half -> 0.5x; blend50 with the CHAIN INPUT -> 0.75x; plus1 -> 0.75x + 0.01
    before, out, info = run("half>blend50>plus1")
    assert np.allclose(out, 0.75 * before + 0.01, atol=1e-6)
    assert info["errors"] == []


def test_failure_falls_back_to_input():
    before, out, info = run("half>blend50>boom")
    assert np.array_equal(out, before) and info["errors"]


def test_leading_and_blend_only_chains_are_dropped():
    for spec, text in (("blend50>half", "first"), ("blend50", "first")):
        before, out, info = run(spec)
        assert np.array_equal(out, before) and FakeBackend.calls == []
        assert info["errors"] and text in info["errors"][0]


def test_unknown_backend_still_fails_but_blend_is_not_unknown():
    before, out, info = run("typo>blend50")
    assert np.array_equal(out, before) and "typo" in info["errors"][0]


def test_json_keys_old_and_new():
    _, _, info = run("half")
    for key in ("chain", "chain_id", "sample_rate", "samples", "lag_samples",
                "noise_floor_db_before", "noise_floor_db_after", "corr_with_input", "errors"):
        assert key in info
    assert info["blend"] is None and len(info["steps"]) == 1
    assert info["steps"][0]["kind"] == "backend"
    _, _, none_info = run("none")
    assert none_info["steps"] == [] and none_info["blend"] is None


def test_nan_and_silence_rejected_after_blend(monkeypatch):
    nan_wav = speech()
    nan_wav[3] = np.nan
    # a NaN chain input leaks through the blend even though the backend output is clean
    data = AudioData(waveform=nan_wav, sample_rate=SR, name="x", audio_segment=None, duration=2.0)
    orig = denoise_service._load_api()

    def clean_backend(name, cfg=None, logger=None):
        b = FakeBackend(name, "identity")
        b.enhance = lambda wav, sr: np.nan_to_num(wav).astype(np.float32)
        return b

    monkeypatch.setattr(denoise_service, "_load_api",
                        lambda: (clean_backend, orig[1], orig[2]))
    info = DenoiseService().apply(data, "identity>blend50", None, None)
    assert info["errors"] and "NaN" in info["errors"][0]
    # a blend that returns silence where the input had speech
    monkeypatch.setattr(pseudo, "blend", lambda c, i, p: np.zeros_like(c))
    info = DenoiseService().apply(audio(), "identity>blend50", None, None)
    assert info["errors"] and "silent" in info["errors"][0]


def test_golden_ids():
    assert chain_id_of("none") == "none"
    assert chain_id_of("dfn3") == "dfn3"
    assert chain_id_of("dfn3>mf2se48") == "dfn3+mf2se48"
    cid = chain_id_of("DFN3>blend50")
    assert cid == "dfn3+blend50" and cid.replace("+", "").isalnum()
