"""数据质量检查脚本。

用法：
    python scripts/check_data.py [--data-dir data]

输出：样本总量、各场景分布、输入/输出长度统计、输出 JSON 数组长度分布、重复率。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _extract_user(record: dict) -> str:
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return record.get("input", "")


def _extract_assistant(record: dict) -> str:
    if "messages" in record:
        return record["messages"][-1].get("content", "")
    return record.get("output", "")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": int(s[0]),
        "max": int(s[-1]),
        "avg": round(sum(s) / n, 1),
        "p50": int(s[n // 2]),
        "p95": int(s[min(int(n * 0.95), n - 1)]),
    }


def check_split(name: str, records: list[dict]) -> None:
    if not records:
        print(f"  {name}: 文件不存在或为空")
        return

    inputs   = [_extract_user(r) for r in records]
    outputs  = [_extract_assistant(r) for r in records]
    scenarios = [r.get("scenario", "") for r in records]

    dup_inputs = len(inputs) - len(set(inputs))

    array_lens: list[int] = []
    parse_fail = 0
    for out in outputs:
        try:
            arr = json.loads(out)
            if isinstance(arr, list):
                array_lens.append(len(arr))
            else:
                parse_fail += 1
        except (json.JSONDecodeError, TypeError):
            parse_fail += 1

    print(f"\n── {name}（{len(records)} 条）──")
    print(f"  重复输入：{dup_inputs} 条")
    print(f"  输出 JSON 解析失败：{parse_fail} 条")

    if any(scenarios):
        sc_counter = Counter(s for s in scenarios if s)
        print("  场景分布：")
        for sc, cnt in sorted(sc_counter.items(), key=lambda x: -x[1]):
            bar = "█" * (cnt * 20 // max(sc_counter.values()))
            print(f"    {sc:<12} {cnt:>4} 条  {bar}")

    in_lens  = [len(t) for t in inputs]
    out_lens = [len(t) for t in outputs]
    print(f"  输入字符长度：{_stats(in_lens)}")
    print(f"  输出字符长度：{_stats(out_lens)}")

    if array_lens:
        arr_counter = Counter(array_lens)
        print(f"  输出数组长度分布：{ dict(sorted(arr_counter.items())) }")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据质量检查脚本")
    parser.add_argument("--data-dir", default="data", help="数据目录（默认 ./data）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"数据目录：{data_dir.resolve()}")

    train = _load_jsonl(data_dir / "train.jsonl")
    val   = _load_jsonl(data_dir / "val.jsonl")

    check_split("train", train)
    check_split("val",   val)

    if train and val:
        total = len(train) + len(val)
        print(f"\n总计：{total} 条  train {len(train)} ({len(train)/total:.0%})  val {len(val)} ({len(val)/total:.0%})")


if __name__ == "__main__":
    main()
