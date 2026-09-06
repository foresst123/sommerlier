"""Running the frame tagger over a recording that does not fit in memory.

panns_inference does one forward pass over whatever it is handed, and the
framewise stack holds an activation per 10ms frame. A 50-minute podcast is 96M
samples at 32kHz; asking for that in one pass wants tens of gigabytes and dies
before it labels anything. The tagger therefore chunks -- and the only thing
worth testing about a chunked pass is that it produces what the single pass
would have produced.

Run:  python -m pytest tests/test_panns_framewise.py -q   (from podcast-pipeline/)
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# panns_inference pulls torchlibrosa and a 312MB checkpoint; neither is needed
# to test how the recording is cut up and put back together.
_fake = types.ModuleType("panns_inference")
_fake.AudioTagging = object
_fake.SoundEventDetection = object
sys.modules.setdefault("panns_inference", _fake)

from models.panns import PANNSDetector


class RecordingSED:
    """Stands in for the model: labels each frame with its own sample index.

    Frame i of a call covers sample i * hop of what that call was fed, so a
    correctly cropped and reassembled result reads 0, hop, 2*hop, ... straight
    through -- and any slip at a chunk boundary shows up as a jump.
    """

    def __init__(self):
        self.calls = []

    def inference(self, batch):
        audio = batch[0]
        self.calls.append(len(audio))
        frames = len(audio) // PANNSDetector.SED_HOP
        out = np.zeros((frames, 527), dtype=np.float32)
        out[:, 0] = audio[::PANNSDetector.SED_HOP][:frames]
        return out[None, :, :]


def _detector(chunk_seconds=1.0, context_seconds=0.25):
    d = PANNSDetector.__new__(PANNSDetector)
    d.device = "cpu"
    d._sed = RecordingSED()
    d.SED_CHUNK_SECONDS = chunk_seconds
    d.SED_CONTEXT_SECONDS = context_seconds
    return d


def _ramp(seconds):
    """Audio whose sample value is its own index, so cropping is checkable."""
    return np.arange(int(seconds * 32000), dtype=np.float32)


def test_a_short_recording_goes_through_in_one_pass():
    d = _detector(chunk_seconds=10.0)
    frames = d._sed_framewise(_ramp(2.0))
    assert len(d._sed.calls) == 1
    assert len(frames) == 200


def test_a_long_recording_is_split():
    d = _detector(chunk_seconds=1.0)
    d._sed_framewise(_ramp(5.0))
    assert len(d._sed.calls) == 5


def test_the_chunks_reassemble_into_the_single_pass_result():
    """The point of the whole exercise: same frames, less memory."""
    audio = _ramp(5.0)
    chunked = _detector(chunk_seconds=1.0)._sed_framewise(audio)
    whole = _detector(chunk_seconds=100.0)._sed_framewise(audio)
    assert len(chunked) == len(whole)
    np.testing.assert_array_equal(chunked[:, 0], whole[:, 0])


def test_every_frame_lands_at_its_own_timestamp():
    """A half-frame slip per chunk would drift into a real timing error."""
    frames = _detector(chunk_seconds=1.0)._sed_framewise(_ramp(7.0))
    expected = np.arange(len(frames), dtype=np.float32) * PANNSDetector.SED_HOP
    np.testing.assert_array_equal(frames[:, 0], expected)


def test_context_is_fed_and_then_thrown_away():
    """Chunks are fed wider than they are kept, or edge frames see silence."""
    d = _detector(chunk_seconds=1.0, context_seconds=0.25)
    d._sed_framewise(_ramp(5.0))
    chunk_samples = 32000
    assert d._sed.calls[1] > chunk_samples, "middle chunk was fed no context"
    assert all(n <= chunk_samples + 2 * 8000 for n in d._sed.calls)


def test_a_short_tail_still_gets_a_second_of_audio():
    """Below a second the pooling stack has nothing to reduce and the call dies."""
    d = _detector(chunk_seconds=1.0, context_seconds=0.0)
    d._sed_framewise(_ramp(2.1))
    assert min(d._sed.calls) >= 32000


def test_a_recording_that_divides_exactly_has_no_extra_chunk():
    d = _detector(chunk_seconds=1.0)
    frames = d._sed_framewise(_ramp(3.0))
    assert len(d._sed.calls) == 3
    assert len(frames) == 300


@pytest.mark.parametrize("seconds", [1.5, 2.0, 3.7, 4.0])
def test_no_frames_are_lost_or_duplicated(seconds):
    frames = _detector(chunk_seconds=1.0)._sed_framewise(_ramp(seconds))
    assert len(frames) == int(seconds * 32000) // PANNSDetector.SED_HOP
