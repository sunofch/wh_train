"""WorkOrder 领域定义的单一真源。

SYSTEM_PROMPT / 字段集合 / 合法枚举 / 奖励权重统一在此定义，
避免在 generate / inference / evaluate 中各抄一份导致漂移。
"""
from __future__ import annotations

# 训练与推理共用的系统提示词
SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)

# WorkOrder 的六个合法字段（顺序即输出顺序）
FIELDS: tuple[str, ...] = (
    "part_name", "quantity", "model",
    "action_required", "is_urgent", "description",
)

# action_required 的合法取值（None 表示无法确定）
VALID_ACTIONS: set[str | None] = {"入库", "出库", "调库", None}

# GRPO 奖励中各字段的权重（体现业务语义重心，和为 1）
FIELD_WEIGHTS: dict[str, float] = {
    "action_required": 0.30,
    "is_urgent":       0.25,
    "part_name":       0.25,
    "quantity":        0.10,
    "model":           0.10,
}
