"""GRPO 标量奖励函数。复用 reward.parser，与评估同源。

reward = 0.10*format + 0.15*schema + 0.15*length + 0.60*fields
各项 ∈ [0,1]；解析失败硬门控为 0。
"""
from __future__ import annotations

from wh_train.schema import FIELDS, VALID_ACTIONS, FIELD_WEIGHTS
from wh_train.reward.parser import parse_orders

_W_FORMAT, _W_SCHEMA, _W_LENGTH, _W_FIELDS = 0.10, 0.15, 0.15, 0.60
_ALLOWED = set(FIELDS)


def _as_text(value) -> str:
    """Normalize TRL chat completions or plain strings to assistant text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                content = item.get("content")
                return content if isinstance(content, str) else ""
        if value:
            return _as_text(value[-1])
    return ""


def _schema_score(orders: list[dict]) -> float:
    """合法工单占比：仅含允许字段 + action 合法 + is_urgent 是 bool。"""
    if not orders:
        return 0.0
    ok = 0
    for o in orders:
        if set(o.keys()) <= _ALLOWED \
                and o.get("action_required") in VALID_ACTIONS \
                and isinstance(o.get("is_urgent"), bool):
            ok += 1
    return ok / len(orders)


def _length_score(pred: list[dict], gold: list[dict]) -> float:
    """长度匹配：相等 1，差 k 线性衰减。"""
    if not gold:
        return 0.0
    diff = abs(len(pred) - len(gold))
    return max(0.0, 1.0 - diff / len(gold))


def _order_field_score(p: dict, g: dict) -> float:
    """单对工单的加权字段命中（null==null 计正确）。"""
    return sum(w for f, w in FIELD_WEIGHTS.items() if p.get(f) == g.get(f))


def _match_pairs(pred: list[dict], gold: list[dict], align: str) -> list[tuple[dict, dict]]:
    """配对 pred 与 gold 工单。positional 按位；greedy 按 part_name 贪心。"""
    if align == "positional":
        return list(zip(pred, gold))
    # greedy：每个 gold 从 pred 池挑 part_name 相等者优先，否则第一个
    pool = list(pred)
    pairs = []
    for g in gold:
        match = next((p for p in pool if p.get("part_name") == g.get("part_name")), None)
        if match is None and pool:
            match = pool[0]
        if match is not None:
            pool.remove(match)
            pairs.append((match, g))
    return pairs


def _fields_score(pred: list[dict], gold: list[dict], align: str) -> float:
    """对齐工单的加权字段命中，按 gold 工单数平均。"""
    if not gold:
        return 0.0
    pairs = _match_pairs(pred, gold, align)
    total = sum(_order_field_score(p, g) for p, g in pairs)
    return total / len(gold)


def compute_reward(pred_text: str, gold_text: str, *, align: str = "positional") -> float:
    """计算单条生成的奖励 ∈ [0,1]。解析失败返回 0。"""
    pred = parse_orders(pred_text)
    gold = parse_orders(gold_text)
    if pred is None or gold is None:
        return 0.0
    r_schema = _schema_score(pred)
    r_length = _length_score(pred, gold)
    r_fields = _fields_score(pred, gold, align)
    return _W_FORMAT * 1.0 + _W_SCHEMA * r_schema + _W_LENGTH * r_length + _W_FIELDS * r_fields


def reward_func(prompts=None, completions=None, gold=None, align: str = "positional", **kwargs) -> list[float]:
    """trl GRPOTrainer 奖励回调。

    completions: list[str]（模型生成）
    gold: list[str]（数据集透传的标准答案列）
    返回每条 completion 的标量奖励。
    """
    completions = completions or []
    gold = gold or []
    return [compute_reward(_as_text(c), _as_text(g), align=align) for c, g in zip(completions, gold)]
