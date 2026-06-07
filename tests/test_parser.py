from wh_train.reward.parser import (
    parse_orders,
    extract_output,
    extract_input,
)

_VALID = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'


def test_parse_orders_valid_array():
    orders = parse_orders(_VALID)
    assert isinstance(orders, list)
    assert orders[0]["part_name"] == "轴承"


def test_parse_orders_invalid_returns_none():
    assert parse_orders("not json") is None


def test_parse_orders_empty_array_returns_none():
    assert parse_orders("[]") is None


def test_parse_orders_strips_markdown_fence():
    fenced = "```json\n" + _VALID + "\n```"
    orders = parse_orders(fenced)
    assert orders is not None
    assert orders[0]["action_required"] == "出库"


def test_extract_output_from_messages():
    rec = {"messages": [
        {"role": "user", "content": "出库2个轴承"},
        {"role": "assistant", "content": _VALID},
    ]}
    assert extract_output(rec, prediction=True) == _VALID


def test_extract_output_from_predict_field():
    rec = {"predict": _VALID, "label": _VALID}
    assert extract_output(rec, prediction=True) == _VALID
    assert extract_output(rec, prediction=False) == _VALID


def test_extract_input_from_messages():
    rec = {"messages": [{"role": "user", "content": "出库2个轴承"}]}
    assert extract_input(rec) == "出库2个轴承"
