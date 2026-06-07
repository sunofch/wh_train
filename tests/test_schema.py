from wh_train.schema import (
    SYSTEM_PROMPT,
    FIELDS,
    VALID_ACTIONS,
    FIELD_WEIGHTS,
)


def test_fields_are_the_six_workorder_keys():
    assert FIELDS == (
        "part_name", "quantity", "model",
        "action_required", "is_urgent", "description",
    )


def test_valid_actions_contains_three_actions_and_none():
    assert VALID_ACTIONS == {"入库", "出库", "调库", None}


def test_system_prompt_mentions_null_rule():
    assert "null" in SYSTEM_PROMPT
    assert "action_required" in SYSTEM_PROMPT


def test_field_weights_sum_to_one():
    assert abs(sum(FIELD_WEIGHTS.values()) - 1.0) < 1e-9


def test_field_weights_keys_are_scored_fields():
    assert set(FIELD_WEIGHTS) == {
        "part_name", "quantity", "model", "action_required", "is_urgent",
    }
