"""数据集划分、去重、格式转换、知识库读取。"""
from __future__ import annotations

import random
from pathlib import Path


def to_openai_messages_format(sample: dict, system_prompt: str) -> dict:
    """{input, output} → OpenAI messages 格式，保留 scenario（若存在）。"""
    result = {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }
    if "scenario" in sample:
        result["scenario"] = sample["scenario"]
    return result


def load_knowledge_base(kb_dir: str, max_chars_per_file: int = 500) -> str:
    """读取知识库 markdown，拼接文本；目录不存在返回空串。"""
    kb_path = Path(kb_dir)
    if not kb_path.is_dir():
        return ""
    texts = []
    for f in sorted(kb_path.glob("*.md")):
        texts.append(f.read_text(encoding="utf-8")[:max_chars_per_file])
    return "\n\n".join(texts)


def stratified_split(samples, val_ratio: float = 0.1, seed: int = 42):
    """按场景分层划分 train/val，sample dict 嵌入 scenario 字段。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for scenario, sample in samples:
        groups.setdefault(scenario, []).append({**sample, "scenario": scenario})
    train_all, val_all = [], []
    for _, group in groups.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_all.extend(shuffled[:n_val])
        train_all.extend(shuffled[n_val:])
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    return train_all, val_all


def deduplicate(samples):
    """按 input 文本去重。"""
    seen: set[str] = set()
    result = []
    for scenario, sample in samples:
        if sample["input"] not in seen:
            seen.add(sample["input"])
            result.append((scenario, sample))
    return result
