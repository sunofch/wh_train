from evaluate import (
    array_length_accuracy,
    evaluate_file,
    field_accuracy,
    json_parse_rate,
    null_recall,
)


def _r(pred: str, gold: str) -> dict:
    return {"predicted": pred, "gold": gold}


_VALID = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
_VALID2 = '[{"part_name":"滤芯","quantity":1,"model":"HC9600","action_required":"入库","is_urgent":false,"description":null}]'
_MULTI = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null},{"part_name":"电机","quantity":1,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
_SINGLE = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'


# ── json_parse_rate ────────────────────────────────────────────────────────────

def test_json_parse_rate_all_valid():
    records = [_r(_VALID, ""), _r(_VALID2, "")]
    assert json_parse_rate(records) == 1.0


def test_json_parse_rate_some_invalid():
    records = [_r("not json", ""), _r(_VALID, "")]
    assert json_parse_rate(records) == 0.5


def test_json_parse_rate_empty_list():
    assert json_parse_rate([]) == 0.0


def test_json_parse_rate_empty_array_is_invalid():
    records = [_r("[]", "")]
    assert json_parse_rate(records) == 0.0


# ── field_accuracy ─────────────────────────────────────────────────────────────

def test_field_accuracy_perfect():
    records = [_r(_VALID, _VALID)]
    assert field_accuracy(records, "part_name") == 1.0


def test_field_accuracy_mismatch():
    pred = '[{"part_name":"轴承","quantity":3,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    gold = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    records = [_r(pred, gold)]
    assert field_accuracy(records, "quantity") == 0.0
    assert field_accuracy(records, "part_name") == 1.0


def test_field_accuracy_empty():
    assert field_accuracy([], "part_name") == 0.0


# ── array_length_accuracy ──────────────────────────────────────────────────────

def test_array_length_accuracy_correct():
    records = [_r(_MULTI, _MULTI), _r(_SINGLE, _SINGLE)]
    assert array_length_accuracy(records) == 1.0


def test_array_length_accuracy_mismatch():
    records = [
        _r(_MULTI, _MULTI),   # 正确（2==2）
        _r(_SINGLE, _MULTI),  # 错误（1!=2）
    ]
    assert array_length_accuracy(records) == 0.5


# ── null_recall ────────────────────────────────────────────────────────────────

def test_null_recall_perfect():
    records = [_r(_VALID, _VALID)]
    assert null_recall(records, "model") == 1.0


def test_null_recall_zero():
    pred = '[{"part_name":"轴承","quantity":2,"model":"6208","action_required":"出库","is_urgent":false,"description":null}]'
    gold = _VALID  # model is null in gold
    records = [_r(pred, gold)]
    assert null_recall(records, "model") == 0.0


def test_null_recall_no_null_cases_returns_one():
    """gold 中该字段全非 null 时，返回 1.0（没有需要召回的 null）。"""
    records = [_r(_VALID2, _VALID2)]
    assert null_recall(records, "model") == 1.0


# ── evaluate_file ──────────────────────────────────────────────────────────────

def test_evaluate_file(tmp_path):
    import json

    system = "你是助手。"
    pred_line = json.dumps({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": "出库2个轴承"},
        {"role": "assistant", "content": _VALID},
    ]}, ensure_ascii=False)
    gold_line = pred_line  # pred == gold → 所有指标为 1.0

    pred_f = tmp_path / "pred.jsonl"
    gold_f = tmp_path / "gold.jsonl"
    pred_f.write_text(pred_line + "\n", encoding="utf-8")
    gold_f.write_text(gold_line + "\n", encoding="utf-8")

    metrics = evaluate_file(str(pred_f), str(gold_f))
    assert metrics["json_parse_rate"] == 1.0
    assert metrics["part_name_accuracy"] == 1.0
    assert metrics["array_length_accuracy"] == 1.0
    assert "model_null_recall" in metrics
