"""Measure separation quality on mixtures simulated from the user's own audio.

SI-SDR/SIR/SAR need reference signals, which a real podcast does not provide.
This builds them: take a stretch where speaker A talks alone, overlay a stretch
where speaker B talks alone at a known level and duration, and the two clean
sources become the references.

The sweep deliberately reproduces the shape that failed in production -- a short,
quiet backchannel buried inside a long turn -- so the numbers say where Sidon
stops working rather than how it does on balanced mixtures it was trained on.

It also records the ECAPA similarity the pipeline's QC actually gates on, so the
two can be correlated: that is what turns TSE_QC_SIM_THRESHOLD from a guess into
a measurement.

Caveat worth reading before trusting the numbers: Sidon resynthesises speech
through a VAE decoder, and SI-SDR is a waveform metric. Perceptually fine output
can score badly. Read SIR (how much of the other speaker leaked) and the ECAPA
column alongside it, and trust relative trends across rows more than absolute
values.

Usage:
    python tools/eval_separation.py --audio /path/to.mp3 \\
        --diarization cache/default_job_<name>/diarization/result.pkl \\
        --out eval_separation.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 24000


# --- metrics -------------------------------------------------------------

def _align(est: np.ndarray, ref: np.ndarray, max_shift: int) -> np.ndarray:
    """Undo the time offset a resynthesising decoder can introduce.

    Without this, SI-SDR punishes a few milliseconds of latency as if it were
    separation error.
    """
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    if max_shift <= 0:
        return est
    corr = np.correlate(est, ref, mode="full")
    centre = len(est) - 1
    lo, hi = centre - max_shift, centre + max_shift + 1
    shift = int(np.argmax(np.abs(corr[lo:hi]))) - max_shift
    if shift > 0:
        return np.concatenate([est[shift:], np.zeros(shift, dtype=est.dtype)])
    if shift < 0:
        return np.concatenate([np.zeros(-shift, dtype=est.dtype), est[:shift]])
    return est


def bss_metrics(est: np.ndarray, target: np.ndarray, interferer: np.ndarray):
    """Scale-invariant SDR / SIR / SAR in dB.

    Decomposition projects the estimate onto the span of the sources:
      e_target = component along the target
      e_interf = the rest of what lies in the source span (the other speaker)
      e_artif  = whatever lies outside it (what the model invented)
    """
    n = min(len(est), len(target), len(interferer))
    est, target, interferer = est[:n], target[:n], interferer[:n]
    est = est - est.mean()

    S = np.stack([target - target.mean(), interferer - interferer.mean()], axis=1)
    if np.linalg.norm(S) < 1e-9 or np.linalg.norm(est) < 1e-9:
        return float("nan"), float("nan"), float("nan")

    # Least-squares projection of est onto span{target, interferer}.
    coef, *_ = np.linalg.lstsq(S, est, rcond=None)
    proj = S @ coef

    t = S[:, 0]
    e_target = (np.dot(est, t) / (np.dot(t, t) + 1e-12)) * t
    e_interf = proj - e_target
    e_artif = est - proj

    CEILING_DB = 100.0

    def db(num, den):
        num, den = float(np.sum(num ** 2)), float(np.sum(den ** 2))
        if num < 1e-12:
            return -CEILING_DB          # nothing of the target survived
        if den < 1e-12:
            return CEILING_DB           # nothing left over: report a ceiling, not nan
        return float(np.clip(10.0 * np.log10(num / den), -CEILING_DB, CEILING_DB))

    sdr = db(e_target, e_interf + e_artif)
    sir = db(e_target, e_interf)
    sar = db(proj, e_artif)
    return sdr, sir, sar


# --- mixture construction ------------------------------------------------

def solo_spans(segments):
    """Per-speaker stretches where nobody else is talking."""
    by_spk = {}
    for s in segments:
        by_spk.setdefault(s.speaker, []).append((s.start, s.end))
    out = {}
    for spk, own in by_spk.items():
        others = sorted((a, b) for o, iv in by_spk.items() if o != spk for a, b in iv)
        spans = []
        for a, b in sorted(own):
            cur = a
            for oa, ob in others:
                if ob <= cur:
                    continue
                if oa >= b:
                    break
                if cur < oa:
                    spans.append((cur, min(oa, b)))
                cur = max(cur, ob)
                if cur >= b:
                    break
            if cur < b:
                spans.append((cur, b))
        out[spk] = [(a, b) for a, b in spans if b - a > 0.5]
    return out


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def build_mixture(host, guest, guest_dur, snr_db, window_sec=20.0):
    """Place `guest` inside `host` at the given duration and level.

    Returns (mixture, ref_host, ref_guest, core_slice).
    """
    n = int(window_sec * SR)
    host = host[:n] if len(host) >= n else np.pad(host, (0, n - len(host)))

    g_n = int(guest_dur * SR)
    guest = guest[:g_n] if len(guest) >= g_n else np.pad(guest, (0, g_n - len(guest)))
    guest = guest * (rms(host) / (rms(guest) + 1e-12)) * (10.0 ** (snr_db / 20.0))

    start = (n - g_n) // 2
    ref_guest = np.zeros(n, dtype=np.float32)
    ref_guest[start:start + g_n] = guest

    mixture = (host + ref_guest).astype(np.float32)
    peak = np.abs(mixture).max()
    if peak > 0.99:
        scale = 0.99 / peak
        mixture, host, ref_guest = mixture * scale, host * scale, ref_guest * scale

    return mixture, host.astype(np.float32), ref_guest, slice(start, start + g_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--diarization", required=True,
                    help="cache/<job>/diarization/result.pkl from a previous run")
    ap.add_argument("--out", default="eval_separation.csv")
    ap.add_argument("--trials", type=int, default=3, help="mixtures per condition")
    ap.add_argument("--durations", default="0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--snrs", default="0,-6,-12,-18", help="guest level vs host, dB")
    args = ap.parse_args()

    import pickle
    import librosa

    with open(args.diarization, "rb") as f:
        diar = pickle.load(f)
    segments = getattr(diar, "segments", diar)

    wav, _ = librosa.load(args.audio, sr=SR, mono=True)
    spans = solo_spans(segments)
    speakers = sorted(spans, key=lambda s: -sum(b - a for a, b in spans[s]))
    if len(speakers) < 2:
        print("Need two speakers with solo audio; found:", list(spans))
        return
    host_spk, guest_spk = speakers[0], speakers[1]
    print(f"host={host_spk} guest={guest_spk}")

    from models.tse_model import TargetSpeakerExtractor
    from services.sidon_worker_service import SidonWorkerService
    import json
    import types

    config = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
    svc_args = types.SimpleNamespace(tse=True, env=os.environ.get("SOMMELIER_ENV", "kaggle"),
                                     config="config.json")
    sidon = SidonWorkerService(config, svc_args, None)
    sidon.spawn()
    sidon.wait_ready()
    tse = TargetSpeakerExtractor(sidon_process=sidon.process)

    def take(spk, want):
        for a, b in sorted(spans[spk], key=lambda s: -(s[1] - s[0])):
            if b - a >= want:
                return wav[int(a * SR):int((a + want) * SR)].copy()
        return None

    host_pool = take(host_spk, 20.0)
    guest_pool = take(guest_spk, 8.0)
    if host_pool is None or guest_pool is None:
        print("Not enough solo audio to build mixtures")
        sidon.stop()
        return

    enroll_host = [take(host_spk, 6.0)]
    enroll_guest = [take(guest_spk, 6.0)]

    rows = []
    for dur in [float(d) for d in args.durations.split(",")]:
        for snr in [float(s) for s in args.snrs.split(",")]:
            for t in range(args.trials):
                off = t * int(2.0 * SR)
                mix, ref_h, ref_g, core = build_mixture(
                    host_pool, guest_pool[off:off + int(8 * SR)] if off else guest_pool,
                    dur, snr)
                probe_h = [(0, core.start)]          # host alone before the guest
                probe_g = None                        # guest never speaks alone here
                try:
                    est_h, est_g, sim_h, sim_g, _diag = tse.separate_two_speakers(
                        mix, enroll_A=enroll_host, enroll_B=enroll_guest,
                        sample_rate=SR, id_A=f"H{t}{dur}{snr}", id_B=f"G{t}{dur}{snr}",
                        probe_A=probe_h, probe_B=probe_g,
                        core_range=(core.start, core.stop))
                except Exception as e:
                    print(f"dur={dur} snr={snr} trial={t}: FAILED {e}")
                    continue

                shift = int(0.05 * SR)
                sdr_h, sir_h, sar_h = bss_metrics(_align(est_h, ref_h, shift), ref_h, ref_g)
                gc = slice(core.start, core.stop)
                sdr_g, sir_g, sar_g = bss_metrics(
                    _align(est_g[gc], ref_g[gc], shift), ref_g[gc], ref_h[gc])

                rows.append(dict(duration=dur, snr_db=snr, trial=t,
                                 host_sdr=sdr_h, host_sir=sir_h, host_sar=sar_h,
                                 guest_sdr=sdr_g, guest_sir=sir_g, guest_sar=sar_g,
                                 ecapa_host=sim_h, ecapa_guest=sim_g))
                print(f"dur={dur:4.2f}s snr={snr:+5.0f}dB  "
                      f"host SDR={sdr_h:6.2f} SIR={sir_h:6.2f}  "
                      f"guest SDR={sdr_g:6.2f} SIR={sir_g:6.2f}  "
                      f"ecapa_host={sim_h}")

    sidon.stop()
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
