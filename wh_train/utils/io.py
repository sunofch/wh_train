"""jsonl 读写工具。"""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    """读取 jsonl，跳过空行。"""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def write_jsonl(path, rows) -> None:
    """写出 jsonl（UTF-8，不转义非 ASCII）。"""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
