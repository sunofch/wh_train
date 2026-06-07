"""评估主流程：读取 pred/gold jsonl，输出指标、按场景细分、错误样本。"""
from __future__ import annotations

import json
from pathlib import Path

from wh_train.reward.parser import parse_orders, extract_output, extract_input
from wh_train.reward.metrics import compute_metrics, evaluate_by_scenario


def _find_errors(records: list[dict]) -> list[dict]:
    """返回解析失败或关键字段不匹配的记录，附 errors 说明。"""
    errors = []
    for r in records:
        pred = parse_orders(r["predicted"])
        gold = parse_orders(r["gold"])
        issues: list[str] = []
        if not pred:
            issues.append("json_parse_failed")
        else:
            if not gold or len(pred) != len(gold):
                issues.append("array_length_mismatch")
            if gold:
                p0, g0 = pred[0], gold[0]
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
    """读取两个 jsonl 计算指标；可选写报告与错误样本。"""
    preds = [json.loads(l) for l in Path(pred_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    golds = [json.loads(l) for l in Path(gold_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]

    records = []
    for p, g in zip(preds, golds):
        r: dict = {
            "predicted": extract_output(p, prediction=True),
            "gold":      extract_output(g, prediction=False),
        }
        scenario = g.get("scenario") or p.get("scenario")
        if scenario:
            r["scenario"] = scenario
        inp = extract_input(g) or extract_input(p)
        if inp:
            r["input"] = inp
        records.append(r)

    overall = compute_metrics(records)
    result: dict = {k: v for k, v in overall.items() if k != "n"}

    if any("scenario" in r for r in records):
        result["by_scenario"] = evaluate_by_scenario(records)

    error_records = _find_errors(records)
    result["error_count"] = len(error_records)

    if report_path:
        Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入：{report_path}")
    if errors_path and error_records:
        with open(errors_path, "w", encoding="utf-8") as f:
            for rec in error_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"错误样本已写入：{errors_path}（共 {len(error_records)} 条）")

    return result


def print_results(metrics: dict) -> None:
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
            print(f"  [{scenario}] n={m.get('n','?')}  parse={m.get('json_parse_rate','?')}  "
                  f"action={m.get('action_accuracy','?')}  urgent={m.get('is_urgent_accuracy','?')}")
