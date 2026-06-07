"""微调模型评估脚本。

用法：
    python evaluate.py pred.jsonl gold.jsonl
    python evaluate.py pred.jsonl gold.jsonl --report report.json --errors errors.jsonl

两个 jsonl 文件可以是 OpenAI messages 格式，或 LLaMA-Factory
generated_predictions.jsonl 格式（predict/label 字段）。
gold.jsonl 若含 scenario 字段（由新版 generate_dataset.py 写入），
则自动输出按场景细分的评估结果。
"""
from __future__ import annotations

import argparse
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


def _extract_input(record: dict) -> str | None:
    """从 messages 格式或 input 字段提取用户输入文本。"""
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return record.get("input")


def _compute_metrics(records: list[dict]) -> dict:
    """计算一组 records 的全部评估指标。"""
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
    """按 scenario 字段分组计算指标，返回 {scenario: metrics}。"""
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = r.get("scenario", "unknown")
        groups.setdefault(key, []).append(r)
    return {scenario: _compute_metrics(group) for scenario, group in sorted(groups.items())}


def _find_errors(records: list[dict]) -> list[dict]:
    """返回存在解析失败或关键字段不匹配的记录，附带 errors 说明列表。"""
    errors = []
    for r in records:
        pred_list = _try_parse(r["predicted"])
        gold_list = _try_parse(r["gold"])
        issues: list[str] = []

        if not pred_list:
            issues.append("json_parse_failed")
        else:
            if not gold_list or len(pred_list) != len(gold_list):
                issues.append("array_length_mismatch")
            if gold_list:
                p0, g0 = pred_list[0], gold_list[0]
                for field in ("part_name", "action_required", "is_urgent", "quantity", "model"):
                    if p0.get(field) != g0.get(field):
                        issues.append(f"{field}_mismatch")

        if issues:
            errors.append({
                "input":     r.get("input", ""),
                "predicted": r["predicted"],
                "gold":      r["gold"],
                "scenario":  r.get("scenario", ""),
                "errors":    issues,
            })
    return errors


def evaluate_file(
    pred_jsonl: str,
    gold_jsonl: str,
    *,
    report_path: str | None = None,
    errors_path: str | None = None,
) -> dict:
    """从两个 jsonl 文件读取并计算全部指标。

    返回 dict 包含整体指标（顶层键）；若 gold 携带 scenario 字段则额外含
    by_scenario 键。可选将完整报告写入 report_path，错误样本写入 errors_path。
    """
    preds = [json.loads(l) for l in Path(pred_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    golds = [json.loads(l) for l in Path(gold_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]

    records = []
    for p, g in zip(preds, golds):
        r: dict = {
            "predicted": _extract_output(p, prediction=True),
            "gold":      _extract_output(g, prediction=False),
        }
        scenario = g.get("scenario") or p.get("scenario")
        if scenario:
            r["scenario"] = scenario
        inp = _extract_input(g) or _extract_input(p)
        if inp:
            r["input"] = inp
        records.append(r)

    overall = _compute_metrics(records)
    result: dict = {
        "json_parse_rate":       overall["json_parse_rate"],
        "part_name_accuracy":    overall["part_name_accuracy"],
        "quantity_accuracy":     overall["quantity_accuracy"],
        "action_accuracy":       overall["action_accuracy"],
        "is_urgent_accuracy":    overall["is_urgent_accuracy"],
        "array_length_accuracy": overall["array_length_accuracy"],
        "model_null_recall":     overall["model_null_recall"],
        "quantity_null_recall":  overall["quantity_null_recall"],
    }

    if any("scenario" in r for r in records):
        result["by_scenario"] = evaluate_by_scenario(records)

    error_records = _find_errors(records)
    result["error_count"] = len(error_records)

    if report_path:
        Path(report_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"报告已写入：{report_path}")

    if errors_path and error_records:
        with open(errors_path, "w", encoding="utf-8") as f:
            for rec in error_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"错误样本已写入：{errors_path}（共 {len(error_records)} 条）")

    return result


def _print_results(metrics: dict) -> None:
    """格式化打印评估结果。"""
    overall_keys = (
        "json_parse_rate", "part_name_accuracy", "quantity_accuracy",
        "action_accuracy", "is_urgent_accuracy", "array_length_accuracy",
        "model_null_recall", "quantity_null_recall", "error_count",
    )
    print("\n=== 整体指标 ===")
    for k in overall_keys:
        if k in metrics:
            print(f"  {k}: {metrics[k]}")

    if "by_scenario" in metrics:
        print("\n=== 按场景细分 ===")
        for scenario, m in metrics["by_scenario"].items():
            n = m.get("n", "?")
            parse = m.get("json_parse_rate", "?")
            action = m.get("action_accuracy", "?")
            urgent = m.get("is_urgent_accuracy", "?")
            print(f"  [{scenario}] n={n}  parse={parse}  action={action}  urgent={urgent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="微调模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pred_jsonl", help="预测结果 JSONL 文件")
    parser.add_argument("gold_jsonl", help="标准答案 JSONL 文件")
    parser.add_argument("--report", default=None, metavar="PATH", help="将完整报告写为 JSON 文件")
    parser.add_argument("--errors", default=None, metavar="PATH", help="将错误样本写为 JSONL 文件")
    args = parser.parse_args()

    metrics = evaluate_file(
        args.pred_jsonl,
        args.gold_jsonl,
        report_path=args.report,
        errors_path=args.errors,
    )
    _print_results(metrics)
