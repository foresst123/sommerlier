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


# --- dryNN: mix back the input of the preceding backend --------------------

@pytest.mark.parametrize("tok,val", [("dry5", 5), ("dry050", 50), ("dry0", 0), ("dry100", 100), ("DRY30", 30)])
def test_parse_dry(tok, val):
    assert parse_pseudo(tok) == ("dry", val) and is_pseudo(tok)


@pytest.mark.parametrize("tok", ["dry101", "dry", "dryx", "xdry50", "dry-1", "drying50"])
def test_parse_dry_invalid(tok):
    assert parse_pseudo(tok) is None


def test_dry_needs_a_backend_before_it():
    validate_chain_tokens(["a", "dry50"])
    validate_chain_tokens(["a", "blend25", "b", "dry50"])
    with pytest.raises(ValueError, match="cannot be first"):
        validate_chain_tokens(["dry50", "a"])
    assert "preceding backend" in describe("dry50") and backend_tokens(["a", "dry50", "b"]) == ["a", "b"]


def test_dry_after_the_first_backend_equals_blend_of_the_chain_input():
    before, out_dry, _ = run("half>dry50")
    _, out_blend, _ = run("half>blend50")
    assert np.allclose(out_dry, out_blend, atol=1e-7) and np.allclose(out_dry, 0.75 * before, atol=1e-6)


def test_dry_mixes_back_the_input_of_its_own_stage_not_the_raw_audio():
    # half, half: the second stage was given 0.5x and outputs 0.25x; dry50 gives 0.5*0.25x + 0.5*0.5x = 0.375x.
    # blend50 would reach for the raw x instead: 0.5*0.25x + 0.5*x = 0.625x.
    before, out_dry, info = run("half>half>dry50")
    assert np.allclose(out_dry, 0.375 * before, atol=1e-6)
    _, out_blend, _ = run("half>half>blend50")
    assert np.allclose(out_blend, 0.625 * before, atol=1e-6)
    assert info["dry"] == [{"percent": 50}] and info["blend"] is None
    assert [(s["token"], s["kind"]) for s in info["steps"]] == [("half", "backend"), ("half", "backend"), ("dry50", "dry")]


def test_two_stage_chain_with_a_light_first_stage_and_a_light_second_stage():
    # half, blend50 -> 0.75x; second half sees 0.75x and gives 0.375x; dry50 -> 0.5*0.375x + 0.5*0.75x = 0.5625x
    before, out, info = run("half>blend50>half>dry50")
    assert np.allclose(out, 0.5625 * before, atol=1e-6)
    assert info["chain_id"] == "half+blend50+half+dry50" and info["errors"] == []


def test_dry_before_any_backend_is_refused_and_keeps_the_unfiltered_audio():
    before, out, info = run("dry50>half")
    assert np.array_equal(out, before) and info["errors"] and "cannot be first" in info["errors"][0]


# ---- hpNN: high-pass in Hz

@pytest.mark.parametrize("tok,val", [("hp60", ("hp", 60)), ("HP80", ("hp", 80)), ("hp10", ("hp", 10)), ("hp200", ("hp", 200))])
def test_parse_hp(tok, val):
    assert parse_pseudo(tok) == val and is_pseudo(tok)


@pytest.mark.parametrize("tok", ["hp", "hp5", "hp9", "hp201", "hp1000", "hp-60", "hp6.0"])
def test_parse_hp_invalid(tok):
    assert parse_pseudo(tok) is None


def test_hp_may_lead_a_chain_but_blend_may_not():
    validate_chain_tokens(["hp60", "half"])
    validate_chain_tokens(["half", "hp60"])
    with pytest.raises(ValueError):
        validate_chain_tokens(["blend20", "half"])


def test_highpass_removes_rumble_and_keeps_a_low_male_fundamental():
    from models.denoise.pseudo import highpass
    t = np.arange(SR * 3) / SR
    f = lambda hz: np.sin(2 * np.pi * hz * t).astype(np.float32)
    rms = lambda x: float(np.sqrt(np.mean(x[SR // 2:-SR // 2] ** 2)))
    assert rms(highpass(f(30), SR, 60)) < 0.12 * rms(f(30))     # 30 Hz rumble is removed
    kept = rms(highpass(f(85), SR, 60)) / rms(f(85))
    assert 0.76 < kept < 0.84                                   # 85 Hz loses about 1.9 dB (zero-phase squares the response)
    assert rms(highpass(f(300), SR, 60)) > 0.99 * rms(f(300))
    assert len(highpass(f(85), SR, 60)) == len(t)


def test_hp_step_runs_in_a_chain_and_is_recorded():
    a = audio(speech(3.0))
    info = DenoiseService().apply(a, "half>hp60")
    assert info["errors"] == [] and info["highpass_hz"] == 60
    assert [s["kind"] for s in info["steps"]] == ["backend", "hp"]


def test_hp_step_in_apply_many():
    xs = [speech(3.0), speech(2.5)]
    outs, info = DenoiseService().apply_many(xs, SR, "half>hp60")
    assert info["errors"] == [] and info["highpass_hz"] == 60
    assert [len(o) for o in outs] == [len(x) for x in xs]
