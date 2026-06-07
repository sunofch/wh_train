"""GRPO 强化精修训练（trl GRPOTrainer + vLLM rollout）。"""
from __future__ import annotations

from wh_train.schema import SYSTEM_PROMPT
from wh_train.utils.io import read_jsonl


def build_grpo_rows(train_jsonl: str) -> list[dict]:
    """从 messages 格式 jsonl 构建 GRPO 行：{prompt: 消息列表, gold: 标准答案串}。

    GRPO 不需要 assistant 目标作为标签，assistant 内容作为 gold 供奖励函数判分。
    跳过缺少 user 消息的记录。
    """
    rows = []
    for rec in read_jsonl(train_jsonl):
        msgs = rec.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
        if user is None or assistant is None:
            continue
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user["content"]},
            ],
            "gold": assistant["content"],
        })
    return rows
