"""ClearVoice picks a GPU through nvidia-smi, which ignores CUDA_VISIBLE_DEVICES."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.denoise.mf2se48 import _pin_to_visible_gpu  # noqa: E402


def test_the_visible_gpu_is_used_not_the_one_nvidia_smi_names():
    class Net:
        def get_free_gpu(self):
            return 1                      # nvidia-smi's answer; only device 0 is visible

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0))
    _pin_to_visible_gpu(Net, torch)
    assert Net().get_free_gpu() == 0
    torch.cuda.is_available = lambda: False
    assert Net().get_free_gpu() is None
