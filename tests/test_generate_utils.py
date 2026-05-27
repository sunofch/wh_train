import json
import pytest
from pathlib import Path
from generate_dataset import validate_sample, to_openai_messages_format, load_knowledge_base, stratified_split

SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)

# ── validate_sample ────────────────────────────────────────────────────────────

def test_validate_sample_valid():
    sample = {
        "input": "出库2个轴承",
        "output": '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    }
    assert validate_sample(sample) is True

def test_validate_sample_missing_part_name():
    sample = {
        "input": "出库2个",
        "output": '[{"quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    }
    assert validate_sample(sample) is False

def test_validate_sample_invalid_action():
    sample = {
        "input": "出库2个轴承",
        "output": '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"删除","is_urgent":false,"description":null}]'
    }
    assert validate_sample(sample) is False

def test_validate_sample_invalid_json():
    sample = {"input": "x", "output": "not json"}
    assert validate_sample(sample) is False

def test_validate_sample_empty_array():
    sample = {"input": "x", "output": "[]"}
    assert validate_sample(sample) is False

# ── to_openai_messages_format ─────────────────────────────────────────────────

def test_to_openai_messages_format_structure():
    sample = {
        "input": "出库2个轴承",
        "output": '[{"part_name":"轴承","quantity":2,"model":null,"action_required":"出库","is_urgent":false,"description":null}]'
    }
    result = to_openai_messages_format(sample, SYSTEM_PROMPT)
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"
    assert result["messages"][1]["content"] == "出库2个轴承"
    assert result["messages"][2]["content"] == sample["output"]

# ── load_knowledge_base ────────────────────────────────────────────────────────

def test_load_knowledge_base(tmp_path):
    (tmp_path / "01_parts.md").write_text("# 轴承\n型号：6208", encoding="utf-8")
    (tmp_path / "02_electrical.md").write_text("# 电机\n型号：Y160M", encoding="utf-8")
    result = load_knowledge_base(str(tmp_path))
    assert "轴承" in result
    assert "电机" in result

# ── stratified_split ───────────────────────────────────────────────────────────

def test_stratified_split_ratio():
    samples = [("显式", {"input": f"x{i}", "output": "[]"}) for i in range(20)]
    samples += [("隐式", {"input": f"y{i}", "output": "[]"}) for i in range(10)]
    train, val = stratified_split(samples, val_ratio=0.1, seed=42)
    assert len(train) + len(val) == 30
    assert abs(len(val) - 3) <= 1  # 10% of 30

def test_stratified_split_each_class_represented():
    samples = [("类A", {"input": f"a{i}", "output": "[]"}) for i in range(10)]
    samples += [("类B", {"input": f"b{i}", "output": "[]"}) for i in range(10)]
    train, val = stratified_split(samples, val_ratio=0.1, seed=42)
    a_in_val = any(s["input"].startswith("a") for s in val)
    b_in_val = any(s["input"].startswith("b") for s in val)
    assert a_in_val and b_in_val
