"""模型输出解析与记录字段提取。

evaluate 与 reward_fn 共用本模块，确保「训练判分口径 = 评估口径」。
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_orders(text: str) -> list[dict] | None:
    """把模型输出文本解析为非空 WorkOrder 数组；失败或空数组返回 None。

    容错：剥离 markdown 代码围栏后再尝试解析。
    """
    if not isinstance(text, str):
        return None
    candidate = _FENCE_RE.sub("", text).strip()
    try:
        result = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(result, list) and len(result) > 0 and all(isinstance(o, dict) for o in result):
        return result
    return None


def extract_output(record: dict, *, prediction: bool) -> str:
    """从支持的 jsonl 格式中提取模型输出字符串（messages / predict / label 等）。"""
    if "messages" in record:
        return record["messages"][-1]["content"]
    keys = ("predict", "prediction", "generated_text") if prediction else ("label", "gold", "output")
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise KeyError(f"Cannot find {'prediction' if prediction else 'gold'} output in record")


def extract_input(record: dict) -> str | None:
    """从 messages 格式或 input 字段提取用户输入文本。"""
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return record.get("input")
