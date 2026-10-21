from pathlib import Path

import pytest

from minicpmo_train.config import load_config


def test_example_config_is_valid() -> None:
    path = Path(__file__).parents[1] / "configs" / "a100_2x.yaml"
    config = load_config(path)
    assert config.model.name_or_path == "openbmb/MiniCPM-o-4_5"
    assert config.data.sample_rate == 16000
    assert config.train.per_device_batch_size == 1


def test_unknown_key_fails_instead_of_being_ignored(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  surprise: 1\ndata:\n  train_manifest: x\ntrain:\n  output_dir: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys in 'model'"):
        load_config(path)
