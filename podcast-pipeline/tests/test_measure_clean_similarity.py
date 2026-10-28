"""tools/measure_clean_similarity.py with a fake embedder and synthetic exports (no model, no GPU)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sf = pytest.importorskip("soundfile")
import measure_clean_similarity as m  # noqa: E402

SR = 16000


class FakeEmbedder:
    """The embedding is the normalised magnitude spectrum in 8 bands: a 220 Hz and a 330 Hz voice differ."""

    def embed(self, wave, sr):
        spec = np.abs(np.fft.rfft(np.asarray(wave, dtype=np.float64)[: sr * 5]))
        freqs = np.fft.rfftfreq(min(len(wave), sr * 5), 1.0 / sr)
        bands = np.array([spec[(freqs >= lo) & (freqs < lo + 100)].sum() for lo in range(0, 800, 100)])
        return (bands / (np.linalg.norm(bands) + 1e-9)).astype(np.float32)


def _tone(freq, a, b, n, amp=0.3):
    x = np.zeros(n, dtype=np.float32)
    t = np.arange(int((b - a) * SR)) / SR
    x[int(a * SR):int(a * SR) + len(t)] = amp * np.sin(2 * np.pi * freq * t)
    return x


def _export(root, name, clean_user=None, clean_assistant_freq=220, user_freq=330, noise=0.0, left="assistant"):
    d = root / "interrupt" / name
    d.mkdir(parents=True)
    n = 8 * SR
    rng = np.random.default_rng(0)
    assistant = _tone(220, 0.5, 3.5, n) + _tone(220, 5, 7.5, n)
    user = _tone(user_freq, 3.8, 5, n) + _tone(user_freq, 1, 2, n) * 0
    if noise:
        assistant = assistant + (rng.standard_normal(n) * noise).astype(np.float32) * (np.abs(assistant) > 0)
    raw = np.column_stack([assistant, user]) if left == "assistant" else np.column_stack([user, assistant])
    a_clean = assistant * 0.9 if clean_assistant_freq == 220 else _tone(clean_assistant_freq, 0.5, 3.5, n)
    u_clean = user if clean_user is None else clean_user
    clean = np.column_stack([a_clean, u_clean]) if left == "assistant" else np.column_stack([u_clean, a_clean])
    sf.write(str(d / "stereo_2ch.wav"), raw, SR, subtype="PCM_16")
    sf.write(str(d / "stereo_2ch_clean.wav"), clean, SR, subtype="PCM_16")
    meta = {"channels": {"left": left},
            "conversation": [{"role": "assistant", "start": 0.5, "end": 3.5}, {"role": "user", "start": 3.8, "end": 5.0}]}
    (d / "conversation.json").write_text(json.dumps(meta))
    return d


def test_same_voice_after_the_filter_has_a_tiny_distance(tmp_path):
    d = _export(tmp_path, "same")
    row = m.measure_folder(d, FakeEmbedder())
    assert row["assistant_filtered"] == 1 and row["user_filtered"] == 0
    assert row["assistant_spk_dist"] < 0.01
    assert row["assistant_spk_sim"] == pytest.approx(1.0 - row["assistant_spk_dist"], abs=1e-4)


def test_an_untouched_channel_is_the_identity(tmp_path):
    row = m.measure_folder(_export(tmp_path, "id"), FakeEmbedder())
    assert row["user_filtered"] == 0
    assert row["user_spk_dist"] == 0.0 and row["user_si_sdr_db"] == 99.0


def test_a_filter_that_changes_the_voice_has_a_large_distance(tmp_path):
    d = _export(tmp_path, "changed", clean_assistant_freq=550)
    row = m.measure_folder(d, FakeEmbedder())
    assert row["assistant_spk_dist"] > 0.5


def test_cross_distance_is_small_for_one_voice_and_large_for_two(tmp_path):
    two = m.measure_folder(_export(tmp_path, "two", user_freq=450), FakeEmbedder())
    one = m.measure_folder(_export(tmp_path, "one", user_freq=220), FakeEmbedder())
    assert one["cross_dist_raw"] < 0.05 < 0.3 < two["cross_dist_raw"]
    assert one["cross_dist_clean"] <= 0.06


def test_the_channels_follow_meta_left(tmp_path):
    a = m.measure_folder(_export(tmp_path, "l", left="assistant", clean_assistant_freq=550), FakeEmbedder())
    b = m.measure_folder(_export(tmp_path, "r", left="user", clean_assistant_freq=550), FakeEmbedder())
    assert a["assistant_spk_dist"] == pytest.approx(b["assistant_spk_dist"], abs=1e-3)


def test_signal_numbers_without_a_model(tmp_path):
    row = m.measure_folder(_export(tmp_path, "sig", noise=0.01), embedder=None)
    assert "assistant_spk_dist" not in row
    assert row["assistant_level_change_db"] == pytest.approx(-0.9, abs=0.3)      # 0.9x
    assert row["assistant_si_sdr_db"] > 15


def test_main_writes_the_csv_and_prints_the_distances(tmp_path, capsys):
    _export(tmp_path, "a")
    _export(tmp_path, "b", user_freq=220)
    csv_path = tmp_path / "out.csv"
    assert m.main(["--final", str(tmp_path), "--csv", str(csv_path)], embedder_factory=lambda device: FakeEmbedder()) == 0
    text = capsys.readouterr().out
    assert "2 conversations measured" in text and "spk_dist" in text and "cross_dist_raw" in text
    header = csv_path.read_text().splitlines()[0].split(",")
    assert {"assistant_spk_dist", "cross_dist_raw", "user_filtered"} <= set(header)


def test_a_folder_without_the_clean_pair_is_skipped(tmp_path):
    d = _export(tmp_path, "x")
    (d / "stereo_2ch_clean.wav").unlink()
    assert m.measure_folder(d, FakeEmbedder()) is None


def _with_original(tmp_path, orig_sr=SR, changed_voice=False, noise=0.02):
    """An export folder plus the original recording it was cut from (source_start 10 s, source_stem 'ep 01')."""
    audio = tmp_path / "audio"
    audio.mkdir()
    d = _export(tmp_path / "final", "item", clean_assistant_freq=550 if changed_voice else 220)
    meta = json.loads((d / "conversation.json").read_text())
    meta.update(source_stem="ep 01", source_start=10.0, source_end=18.0)
    (d / "conversation.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(1)
    n = int(40 * orig_sr)
    t = np.arange(n) / orig_sr
    orig = (rng.standard_normal(n) * noise).astype(np.float32)
    for a, b, f in ((0.5, 3.5, 220), (5, 7.5, 220), (3.8, 5, 330)):           # item-relative, the same voices
        i, j = int((10 + a) * orig_sr), int((10 + b) * orig_sr)
        orig[i:j] += 0.3 * np.sin(2 * np.pi * f * t[i:j])
    sf.write(str(audio / "ep 01.wav"), orig, orig_sr, subtype="PCM_16")
    return d, audio


def test_the_original_is_compared_with_the_raw_and_the_clean_channel(tmp_path):
    d, audio = _with_original(tmp_path)
    row = m.measure_folder(d, FakeEmbedder(), m.index_originals(audio))
    assert row["original_found"] == 1
    assert row["assistant_orig_vs_clean_dist"] < 0.1                           # the voice is the same all along
    assert row["assistant_floor_orig_dbfs"] < row["assistant_floor_final_dbfs"] + 60
    assert {"assistant_orig_vs_raw_dist", "assistant_raw_vs_clean_dist", "assistant_orig_solo_s"} <= set(row)


def test_a_filter_that_changes_the_voice_shows_in_the_distance_to_the_original(tmp_path):
    d, audio = _with_original(tmp_path, changed_voice=True)
    row = m.measure_folder(d, FakeEmbedder(), m.index_originals(audio))
    assert row["assistant_orig_vs_clean_dist"] > 0.5
    assert row["assistant_orig_vs_raw_dist"] < 0.1


def test_an_original_at_another_sample_rate_is_resampled(tmp_path):
    d, audio = _with_original(tmp_path, orig_sr=32000)
    row = m.measure_folder(d, FakeEmbedder(), m.index_originals(audio))
    assert row["original_found"] == 1 and row["assistant_orig_vs_clean_dist"] < 0.15


def test_the_original_is_found_by_the_sanitised_stem(tmp_path):
    d, audio = _with_original(tmp_path)
    meta = json.loads((d / "conversation.json").read_text())
    meta["source_stem"] = "ep_01"                                              # what the exporter may have written
    (d / "conversation.json").write_text(json.dumps(meta))
    assert m.measure_folder(d, FakeEmbedder(), m.index_originals(audio))["original_found"] == 1


def test_a_missing_original_is_reported_not_fatal(tmp_path):
    d, audio = _with_original(tmp_path)
    (audio / "ep 01.wav").unlink()
    row = m.measure_folder(d, FakeEmbedder(), m.index_originals(audio))
    assert row["original_found"] == 0 and "assistant_orig_vs_clean_dist" not in row


def test_solo_spans_leave_out_the_turns_with_the_other_speaker(tmp_path):
    meta = {"conversation": [{"role": "assistant", "start": 0, "end": 4}, {"role": "user", "start": 3.5, "end": 6},
                             {"role": "assistant", "start": 8, "end": 10}]}
    assert m.solo_spans(meta, "assistant") == [(8.0, 10.0)]
    assert m.solo_spans(meta, "user") == []                                    # overlapped by the first turn


def test_main_prints_the_original_block(tmp_path, capsys):
    d, audio = _with_original(tmp_path)
    assert m.main(["--final", str(tmp_path / "final"), "--original-dir", str(audio), "--csv", str(tmp_path / "o.csv")],
                  embedder_factory=lambda device: FakeEmbedder()) == 0
    text = capsys.readouterr().out
    assert "against the original recording" in text and "orig_vs_clean_dist" in text


def test_exports_not_at_16k_are_resampled_once_before_the_embedder(tmp_path):
    d = _export(tmp_path, "hi")
    for name in ("stereo_2ch.wav", "stereo_2ch_clean.wav"):          # rewrite both at 48 kHz
        x, sr = sf.read(str(d / name), dtype="float32", always_2d=True)
        up = np.column_stack([np.interp(np.linspace(0, len(x) - 1, len(x) * 3), np.arange(len(x)), x[:, k]) for k in (0, 1)])
        sf.write(str(d / name), up.astype(np.float32), 48000, subtype="PCM_16")
    seen = set()

    class Spy(FakeEmbedder):
        def embed(self, wave, sr):
            seen.add(sr)
            return super().embed(wave, sr)

    row = m.measure_folder(d, Spy())
    assert seen == {16000}
    assert row["assistant_spk_dist"] < 0.05 and row["duration_s"] == pytest.approx(8.0, abs=0.1)
