"""[转发壳] 实现已迁移至 wh_train.eval.evaluate / wh_train.reward.*。

保留顶层导入兼容（tests 与既有命令仍可用）。新代码请直接 import wh_train.*。
"""
from __future__ import annotations

import argparse

from wh_train.reward.metrics import (
    json_parse_rate,
    field_accuracy,
    array_length_accuracy,
    null_recall,
)
from wh_train.eval.evaluate import evaluate_file, print_results

__all__ = [
    "json_parse_rate", "field_accuracy", "array_length_accuracy",
    "null_recall", "evaluate_file", "print_results",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调模型评估脚本")
    parser.add_argument("pred_jsonl")
    parser.add_argument("gold_jsonl")
    parser.add_argument("--report", default=None, metavar="PATH")
    parser.add_argument("--errors", default=None, metavar="PATH")
    args = parser.parse_args()
    metrics = evaluate_file(args.pred_jsonl, args.gold_jsonl, report_path=args.report, errors_path=args.errors)
    print_results(metrics)
