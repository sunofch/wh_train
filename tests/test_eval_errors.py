"""覆盖 eval/evaluate 的错误定位、报告输出、按场景细分、打印分支。"""
import json

from wh_train.eval.evaluate import _find_errors, evaluate_file, print_results

_GOLD = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'


def test_find_errors_flags_parse_failure():
    errors = _find_errors([{"predicted": "garbage", "gold": _GOLD, "input": "x"}])
    assert len(errors) == 1
    assert "json_parse_failed" in errors[0]["errors"]


def test_find_errors_flags_field_mismatch():
    pred = '[{"part_name":"电机","quantity":2,"model":null,"action_required":"入库","is_urgent":true,"description":null}]'
    errors = _find_errors([{"predicted": pred, "gold": _GOLD}])
    issues = errors[0]["errors"]
    assert "part_name_mismatch" in issues
    assert "action_required_mismatch" in issues
    assert "is_urgent_mismatch" in issues


def test_find_errors_empty_when_perfect():
    assert _find_errors([{"predicted": _GOLD, "gold": _GOLD}]) == []


def test_evaluate_file_writes_report_and_errors(tmp_path):
    pred_line = json.dumps({"predict": "garbage", "label": _GOLD,
                            "input": "出库2个轴承", "scenario": "显式标准指令"}, ensure_ascii=False)
    gold_line = json.dumps({"label": _GOLD, "scenario": "显式标准指令"}, ensure_ascii=False)
    pred_f = tmp_path / "pred.jsonl"
    gold_f = tmp_path / "gold.jsonl"
    pred_f.write_text(pred_line + "\n", encoding="utf-8")
    gold_f.write_text(gold_line + "\n", encoding="utf-8")
    report = tmp_path / "report.json"
    errors = tmp_path / "errors.jsonl"

    metrics = evaluate_file(str(pred_f), str(gold_f),
                            report_path=str(report), errors_path=str(errors))

    assert metrics["error_count"] == 1
    assert "by_scenario" in metrics
    assert report.is_file()
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["error_count"] == 1
    assert errors.is_file()
    err_rows = [json.loads(l) for l in errors.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert err_rows[0]["scenario"] == "显式标准指令"


def test_print_results_outputs_sections(capsys):
    metrics = {
        "json_parse_rate": 1.0,
        "action_accuracy": 0.9,
        "error_count": 0,
        "by_scenario": {"多备件": {"n": 5, "json_parse_rate": 1.0,
                                   "action_accuracy": 0.8, "is_urgent_accuracy": 0.9}},
    }
    print_results(metrics)
    out = capsys.readouterr().out
    assert "整体指标" in out
    assert "按场景细分" in out
    assert "多备件" in out
