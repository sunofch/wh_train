"""覆盖 cli.main 的分发分支与 train/sft 薄封装（均以 patch 替换重副作用）。"""
from unittest.mock import patch

from wh_train.cli import main
from wh_train.train.sft import run_sft


def test_run_sft_invokes_llamafactory():
    with patch("wh_train.train.sft.subprocess.call", return_value=0) as m:
        rc = run_sft("config/x.yaml")
    assert rc == 0
    m.assert_called_once_with(["llamafactory-cli", "train", "config/x.yaml"])


def test_main_sft_dispatch():
    with patch("wh_train.train.sft.run_sft", return_value=0) as m:
        main(["sft", "--config", "config/sft.yaml"])
    m.assert_called_once_with("config/sft.yaml")


def test_main_gen_data_dispatch():
    with patch("wh_train.data.generate.run", return_value=(10, 1)) as m:
        main(["gen-data", "--output-dir", "out"])
    m.assert_called_once_with("", "out")


def test_main_eval_dispatch(tmp_path, capsys):
    fake_metrics = {"json_parse_rate": 1.0, "error_count": 0}
    with patch("wh_train.eval.evaluate.evaluate_file", return_value=fake_metrics) as m:
        main(["eval", "--pred", "p.jsonl", "--gold", "g.jsonl"])
    m.assert_called_once()
    assert "整体指标" in capsys.readouterr().out


def test_main_grpo_dispatch():
    with patch("wh_train.train.grpo.run_grpo", return_value=None) as m:
        main(["grpo", "--config", "config/grpo.yaml"])
    m.assert_called_once_with("config/grpo.yaml")
