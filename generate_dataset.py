"""DeepSeek API 合成训练数据生成脚本。"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)

VALID_ACTIONS = {"入库", "出库", "调库", None}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", "placeholder"),
            base_url="https://api.deepseek.com",
        )
    return _client

SCENARIOS = {
    "显式标准指令": {
        "count": 300,
        "desc": "指令中明确包含出库/入库/调库/紧急等词汇，字段完整",
    },
    "隐式行为推断": {
        "count": 300,
        "desc": "不含出库/入库/调库，但可从拿/取/要/领/到货/搬移等词推断操作类型",
    },
    "隐式紧急推断": {
        "count": 150,
        "desc": "不含紧急/急，但设备故障/抢修/停产等场景隐含紧急；或明确说下周/不急则非紧急",
    },
    "字段缺失": {
        "count": 225,
        "desc": "缺少型号、数量或操作类型中的一个或多个，���失字段输出null",
    },
    "多备件": {
        "count": 225,
        "desc": "一句话包含多种���件，每种输出一个对象，组成数组",
    },
    "ASR噪声": {
        "count": 150,
        "desc": "包含同音字替换（轴承→周承）、数字口语化（6208→六二零八）、语气词、断句错误等",
    },
    "数量模糊": {
        "count": 150,
        "desc": "用几个/一批/一箱/两三个/大概N个等模糊表达，quantity输出null",
    },
}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def validate_sample(sample: dict) -> bool:
    """验证单条样本是否合法。"""
    try:
        orders = json.loads(sample.get("output", ""))
        if not isinstance(orders, list) or len(orders) == 0:
            return False
        for o in orders:
            if "part_name" not in o:
                return False
            if o.get("action_required") not in VALID_ACTIONS:
                return False
            if not isinstance(o.get("is_urgent"), bool):
                return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def to_swift_format(sample: dict, system_prompt: str) -> dict:
    """将 {input, output} 转为 ms-swift ShareGPT 格式。"""
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": sample["input"]},
            {"role": "assistant", "content": sample["output"]},
        ]
    }


def load_knowledge_base(kb_dir: str, max_chars_per_file: int = 500) -> str:
    """读取知识库 markdown 文件，返回拼接文本。"""
    kb_path = Path(kb_dir)
    texts = []
    for f in sorted(kb_path.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        texts.append(content[:max_chars_per_file])
    return "\n\n".join(texts)


def stratified_split(
    samples: list[tuple[str, dict]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """按场景类别分层划分 train/val，返回 (train_samples, val_samples)。"""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for scenario, sample in samples:
        groups.setdefault(scenario, []).append(sample)

    train_all, val_all = [], []
    for scenario, group in groups.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_all.extend(shuffled[:n_val])
        train_all.extend(shuffled[n_val:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    return train_all, val_all


def generate_batch(
    scenario: str,
    desc: str,
    kb_context: str,
    n: int = 10,
) -> list[dict]:
    """调用 DeepSeek API 生成一批样本，过滤无效结果后返回。"""
    prompt = (
        f"你是港���备件仓储数据标注专家。\n"
        f"知识库摘要（参考真实备件名称和型号）：\n{kb_context}\n\n"
        f"生成 {n} 条【{scenario}】场景训练数据，要求：{desc}\n\n"
        f"返回严格 JSON：\n"
        f'{{ "samples": [ {{"input": "用户指令", "output": "[{{...}}]"}}, ... ] }}\n\n'
        f"output 字段是 WorkOrder JSON 数组字符串，遵守 null 规则，指令表达多样化。"
    )
    resp = _get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        raw = data.get("samples", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [s for s in raw if validate_sample(s)]


def deduplicate(
    samples: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """按 input 文��去重。"""
    seen: set[str] = set()
    result = []
    for scenario, sample in samples:
        key = sample["input"]
        if key not in seen:
            seen.add(key)
            result.append((scenario, sample))
    return result


def main():
    kb_dir = "/home/catlab/wh/wh_graphrag_re/data/knowledge_base"
    output_dir = Path("/home/catlab/wh_train/data")
    output_dir.mkdir(exist_ok=True)

    kb_context = load_knowledge_base(kb_dir)
    all_samples: list[tuple[str, dict]] = []

    for scenario, cfg in SCENARIOS.items():
        print(f"\n生成 {scenario}（目标 {cfg['count']} 条）...")
        collected: list[tuple[str, dict]] = []
        batch_size = 10
        max_iter = cfg["count"] * 3

        for _ in range(0, max_iter, batch_size):
            if len(collected) >= cfg["count"]:
                break
            batch = generate_batch(scenario, cfg["desc"], kb_context, batch_size)
            collected.extend((scenario, s) for s in batch)
            print(f"  已收集 {len(collected)} / {cfg['count']}")

        all_samples.extend(collected[: cfg["count"]])

    all_samples = deduplicate(all_samples)
    train_raw, val_raw = stratified_split(all_samples, val_ratio=0.1, seed=42)

    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_raw:
            f.write(json.dumps(to_swift_format(s, SYSTEM_PROMPT), ensure_ascii=False) + "\n")

    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for s in val_raw:
            f.write(json.dumps(to_swift_format(s, SYSTEM_PROMPT), ensure_ascii=False) + "\n")

    print(f"\n完成：train {len(train_raw)} 条，val {len(val_raw)} 条")
    print(f"输出：{output_dir}/train.jsonl, val.jsonl")


if __name__ == "__main__":
    main()
