"""样本与工单归一化、校验。"""
from __future__ import annotations

import json
from collections import Counter

from wh_train.schema import VALID_ACTIONS
from wh_train.data.scenarios import URGENT_HINTS


def _normalize_action(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if "入库" in text or "到货" in text or "采购" in text:
        return "入库"
    if "出库" in text or "领" in text or "取" in text or "拿" in text:
        return "出库"
    if "调库" in text or "调拨" in text or "移" in text or "搬" in text:
        return "调库"
    return text if text in VALID_ACTIONS else None


def _normalize_urgent(order: dict, instruction: str) -> bool:
    value = order.get("is_urgent")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.lower()
        if low in {"true", "yes", "urgent", "high", "emergency", "紧急", "高"}:
            return True
        if low in {"false", "no", "normal", "low", "普通", "正常", "低"}:
            return False
    priority = order.get("priority")
    if isinstance(priority, str):
        low = priority.lower()
        if low in {"urgent", "high", "emergency", "紧急", "高"}:
            return True
        if low in {"normal", "low", "普通", "正常", "低"}:
            return False
    return any(hint in instruction for hint in URGENT_HINTS)


def _normalize_order(order: object, instruction: str) -> dict:
    if not isinstance(order, dict):
        return {}
    action = order.get("action_required") if "action_required" in order \
        else _normalize_action(order.get("operation", order.get("type")))
    return {
        "part_name": order.get("part_name", order.get("name")),
        "quantity": order.get("quantity"),
        "model": order.get("model"),
        "action_required": action,
        "is_urgent": _normalize_urgent(order, instruction),
        "description": order.get("description", order.get("remark")),
    }


def _normalize_sample(sample: dict) -> dict:
    if not isinstance(sample, dict):
        return {}
    normalized = dict(sample)
    output = normalized.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return normalized
    if isinstance(output, dict):
        output = [output]
    if isinstance(output, list):
        normalized["output"] = json.dumps(
            [_normalize_order(o, normalized.get("input", "")) for o in output],
            ensure_ascii=False,
        )
    return normalized


def _validation_error(sample: dict) -> str | None:
    sample = _normalize_sample(sample)
    if not sample.get("input"):
        return "missing input"
    output = sample.get("output", "")
    if not isinstance(output, str):
        return "output is not json string"
    try:
        orders = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return "output json decode failed"
    if not isinstance(orders, list) or len(orders) == 0:
        return "output is not non-empty array"
    for o in orders:
        if not isinstance(o, dict):
            return "order is not object"
        if not o.get("part_name"):
            return "missing part_name"
        if o.get("action_required") not in VALID_ACTIONS:
            return "invalid action_required"
        if not isinstance(o.get("is_urgent"), bool):
            return "invalid is_urgent"
    return None


def validate_sample(sample: dict) -> bool:
    return _validation_error(sample) is None


def valid_samples(raw: list[dict]) -> list[dict]:
    out = []
    for sample in raw:
        normalized = _normalize_sample(sample)
        if validate_sample(normalized):
            out.append(normalized)
    return out


def rejection_summary(raw: list[dict]) -> str:
    if not raw:
        return "DeepSeek 返回 samples 为空"
    reasons = Counter(_validation_error(s) or "valid" for s in raw)
    return ", ".join(f"{r}: {c}" for r, c in reasons.items())
