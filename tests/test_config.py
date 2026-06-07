import pytest
from wh_train.config import load_yaml


def test_load_yaml_returns_dict(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: Qwen/Qwen3.5-4B\nlearning_rate: 2.0e-6\n", encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg["model"] == "Qwen/Qwen3.5-4B"
    assert cfg["learning_rate"] == 2.0e-6


def test_load_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")
