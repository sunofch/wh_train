from wh_train.reward.metrics import (
    json_parse_rate,
    field_accuracy,
    array_length_accuracy,
    null_recall,
    compute_metrics,
    evaluate_by_scenario,
)

_VALID = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
_MULTI = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null},{"part_name":"电机","quantity":1,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
_SINGLE = _VALID


def _r(pred, gold, scenario=None):
    rec = {"predicted": pred, "gold": gold}
    if scenario:
        rec["scenario"] = scenario
    return rec


def test_json_parse_rate_some_invalid():
    assert json_parse_rate([_r("not json", ""), _r(_VALID, "")]) == 0.5


def test_json_parse_rate_empty_array_invalid():
    assert json_parse_rate([_r("[]", "")]) == 0.0


def test_field_accuracy_mismatch():
    pred = '[{"part_name":"轴承","quantity":3,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    assert field_accuracy([_r(pred, _VALID)], "quantity") == 0.0
    assert field_accuracy([_r(pred, _VALID)], "part_name") == 1.0


def test_array_length_accuracy_mismatch():
    assert array_length_accuracy([_r(_MULTI, _MULTI), _r(_SINGLE, _MULTI)]) == 0.5


def test_null_recall_zero():
    pred = '[{"part_name":"轴承","quantity":2,"model":"6208","action_required":"出库","is_urgent":false,"description":null}]'
    assert null_recall([_r(pred, _VALID)], "model") == 0.0


def test_compute_metrics_has_all_keys():
    m = compute_metrics([_r(_VALID, _VALID)])
    for k in ("n", "json_parse_rate", "part_name_accuracy", "action_accuracy",
              "is_urgent_accuracy", "array_length_accuracy", "model_null_recall"):
        assert k in m


def test_evaluate_by_scenario_groups():
    recs = [_r(_VALID, _VALID, "显式标准指令"), _r(_VALID, _VALID, "多备件")]
    by = evaluate_by_scenario(recs)
    assert set(by) == {"显式标准指令", "多备件"}
    assert by["多备件"]["n"] == 1
