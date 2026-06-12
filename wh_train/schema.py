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

# 基座模型评估专用提示词（字段说明更详细，帮助未微调模型理解格式）
BASE_SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。将用户的自然语言备件操作指令解析为JSON数组，每个备件一个对象。\n\n"
    "每个对象必须包含以下字段：\n"
    "- part_name: 备件名称（字符串，无法确定则为null）\n"
    "- quantity: 数量（整数，无法确定则为null）\n"
    "- model: 型号规格（字符串，无法确定则为null）\n"
    "- action_required: 操作类型，只能是 \"入库\"/\"出库\"/\"调库\"/null 之一\n"
    "- is_urgent: 是否紧急（true/false，默认false）\n"
    "- description: 补充说明（字符串，无则为null）\n\n"
    "规则：\n"
    "1. 纯查询/确认类（无操作意图）返回空数组 []\n"
    "2. 无法确定的字段输出 null，不要猜测\n"
    "3. 只输出JSON数组，不要任何解释文字"
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
