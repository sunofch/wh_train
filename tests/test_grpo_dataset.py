from wh_train.train.grpo import build_grpo_rows

_GOLD = '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'


def test_build_grpo_rows_extracts_prompt_and_gold(tmp_path):
    import json
    line = json.dumps({"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "出库2个轴承"},
        {"role": "assistant", "content": _GOLD},
    ]}, ensure_ascii=False)
    p = tmp_path / "train.jsonl"
    p.write_text(line + "\n", encoding="utf-8")

    rows = build_grpo_rows(str(p))
    assert len(rows) == 1
    assert rows[0]["gold"] == _GOLD
    # prompt 为 system+user 的消息列表（供 trl apply_chat_template）
    assert rows[0]["prompt"][0]["role"] == "system"
    assert rows[0]["prompt"][1]["content"] == "出库2个轴承"


def test_build_grpo_rows_skips_records_without_user(tmp_path):
    import json
    bad = json.dumps({"messages": [{"role": "assistant", "content": _GOLD}]}, ensure_ascii=False)
    p = tmp_path / "train.jsonl"
    p.write_text(bad + "\n", encoding="utf-8")
    assert build_grpo_rows(str(p)) == []
