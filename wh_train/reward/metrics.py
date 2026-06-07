"""评估指标。逻辑复用 reward.parser，保证与训练奖励同源。"""
from __future__ import annotations

from wh_train.reward.parser import parse_orders


def _orders(text: str) -> list[dict]:
    """解析为 orders，失败返回空列表（指标语义下空=解析失败）。"""
    return parse_orders(text) or []


def json_parse_rate(records: list[dict]) -> float:
    """能解析为非空数组的比例。"""
    if not records:
        return 0.0
    ok = sum(1 for r in records if _orders(r["predicted"]))
    return ok / len(records)


def field_accuracy(records: list[dict], field: str) -> float:
    """第一个对象上指定字段与 gold 一致的比例。"""
    if not records:
        return 0.0
    correct = total = 0
    for r in records:
        pred, gold = _orders(r["predicted"]), _orders(r["gold"])
        total += 1
        if not pred or not gold:
            continue
        if pred[0].get(field) == gold[0].get(field):
            correct += 1
    return correct / total if total else 0.0


def array_length_accuracy(records: list[dict]) -> float:
    """数组长度与 gold 一致的比例。"""
    if not records:
        return 0.0
    correct = sum(1 for r in records if len(_orders(r["predicted"])) == len(_orders(r["gold"])))
    return correct / len(records)


def null_recall(records: list[dict], field: str) -> float:
    """gold 该字段为 null 时 pred 也为 null 的比例。"""
    null_cases = [r for r in records if _orders(r["gold"]) and _orders(r["gold"])[0].get(field) is None]
    if not null_cases:
        return 1.0
    correct = sum(
        1 for r in null_cases
        if _orders(r["predicted"]) and _orders(r["predicted"])[0].get(field) is None
    )
    return correct / len(null_cases)


def compute_metrics(records: list[dict]) -> dict:
    """计算一组 records 的全部指标。"""
    return {
        "n": len(records),
        "json_parse_rate":       round(json_parse_rate(records), 4),
        "part_name_accuracy":    round(field_accuracy(records, "part_name"), 4),
        "quantity_accuracy":     round(field_accuracy(records, "quantity"), 4),
        "action_accuracy":       round(field_accuracy(records, "action_required"), 4),
        "is_urgent_accuracy":    round(field_accuracy(records, "is_urgent"), 4),
        "array_length_accuracy": round(array_length_accuracy(records), 4),
        "model_null_recall":     round(null_recall(records, "model"), 4),
        "quantity_null_recall":  round(null_recall(records, "quantity"), 4),
    }


def evaluate_by_scenario(records: list[dict]) -> dict[str, dict]:
    """按 scenario 分组计算指标。"""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("scenario", "unknown"), []).append(r)
    return {sc: compute_metrics(g) for sc, g in sorted(groups.items())}
