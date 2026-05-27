"""微调模型评估脚本。

用法：
    python evaluate.py pred.jsonl gold.jsonl

两个 jsonl 文件可以是 OpenAI messages 格式，或 LLaMA-Factory
generated_predictions.jsonl 格式（predict/label 字段）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _try_parse(output: str) -> list[dict]:
    try:
        result = json.loads(output)
        if isinstance(result, list) and len(result) > 0:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def json_parse_rate(records: list[dict]) -> float:
    """JSON 解析成功率（能解析为非空数组的比例）。"""
    if not records:
        return 0.0
    ok = sum(1 for r in records if _try_parse(r["predicted"]))
    return ok / len(records)


def field_accuracy(records: list[dict], field: str) -> float:
    """指定字段在第一个对象中与 gold 一致的比例。"""
    if not records:
        return 0.0
    correct = 0
    total = 0
    for r in records:
        pred_list = _try_parse(r["predicted"])
        gold_list = _try_parse(r["gold"])
        if not pred_list or not gold_list:
            total += 1
            continue
        if pred_list[0].get(field) == gold_list[0].get(field):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def array_length_accuracy(records: list[dict]) -> float:
    """数组长度与 gold 一致的比例（多备件场景关键指标）。"""
    if not records:
        return 0.0
    correct = sum(
        1 for r in records
        if len(_try_parse(r["predicted"])) == len(_try_parse(r["gold"]))
    )
    return correct / len(records)


def null_recall(records: list[dict], field: str) -> float:
    """gold 为 null 的字段，pred 也输出 null 的比例（衡量不乱猜能力）。"""
    null_cases = [
        r for r in records
        if _try_parse(r["gold"]) and _try_parse(r["gold"])[0].get(field) is None
    ]
    if not null_cases:
        return 1.0
    correct = sum(
        1 for r in null_cases
        if _try_parse(r["predicted"]) and _try_parse(r["predicted"])[0].get(field) is None
    )
    return correct / len(null_cases)


def _extract_output(record: dict, *, prediction: bool) -> str:
    """Extract model output from supported prediction/gold jsonl formats."""
    if "messages" in record:
        return record["messages"][-1]["content"]

    keys = ("predict", "prediction", "generated_text") if prediction else ("label", "gold", "output")
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value

    raise KeyError(f"Cannot find {'prediction' if prediction else 'gold'} output in record")


def evaluate_file(pred_jsonl: str, gold_jsonl: str) -> dict:
    """从两个 jsonl 文件读取并计算全部指标。"""
    preds = [json.loads(l) for l in Path(pred_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    golds = [json.loads(l) for l in Path(gold_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [
        {"predicted": _extract_output(p, prediction=True), "gold": _extract_output(g, prediction=False)}
        for p, g in zip(preds, golds)
    ]
    return {
        "json_parse_rate":       round(json_parse_rate(records), 4),
        "part_name_accuracy":    round(field_accuracy(records, "part_name"), 4),
        "quantity_accuracy":     round(field_accuracy(records, "quantity"), 4),
        "action_accuracy":       round(field_accuracy(records, "action_required"), 4),
        "is_urgent_accuracy":    round(field_accuracy(records, "is_urgent"), 4),
        "array_length_accuracy": round(array_length_accuracy(records), 4),
        "model_null_recall":     round(null_recall(records, "model"), 4),
        "quantity_null_recall":  round(null_recall(records, "quantity"), 4),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py pred.jsonl gold.jsonl")
        sys.exit(1)
    metrics = evaluate_file(sys.argv[1], sys.argv[2])
    for k, v in metrics.items():
        print(f"{k}: {v}")
