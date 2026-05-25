"""DeepSeek API 合成训练数据生成脚本。"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)

VALID_ACTIONS = {"入库", "出库", "调库", None}
URGENT_HINTS = ("紧急", "急需", "抢修", "故障", "停机", "停产", "待修", "马上", "立即")

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
        "desc": "缺少型号、数量或操作类型中的一个或多个，缺失字段输出null",
    },
    "多备件": {
        "count": 225,
        "desc": "一句话包含多种备件，每种输出一个对象，组成数组",
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

def _normalize_sample(sample: dict) -> dict:
    """兼容模型字段别名，并统一为训练目标 schema。"""
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


def _normalize_action(value: object) -> str | None:
    """将常见操作字段别名归一到 action_required。"""
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
    """将 priority 等字段归一为 is_urgent。"""
    value = order.get("is_urgent")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "urgent", "high", "emergency", "紧急", "高"}:
            return True
        if lowered in {"false", "no", "normal", "low", "普通", "正常", "低"}:
            return False

    priority = order.get("priority")
    if isinstance(priority, str):
        lowered = priority.lower()
        if lowered in {"urgent", "high", "emergency", "紧急", "高"}:
            return True
        if lowered in {"normal", "low", "普通", "正常", "低"}:
            return False

    return any(hint in instruction for hint in URGENT_HINTS)


def _normalize_order(order: object, instruction: str) -> dict:
    """将单个工单对象归一为模型输出目标字段。"""
    if not isinstance(order, dict):
        return {}
    if "action_required" in order:
        action = order.get("action_required")
    else:
        action = _normalize_action(order.get("operation", order.get("type")))
    return {
        "part_name": order.get("part_name", order.get("name")),
        "quantity": order.get("quantity"),
        "model": order.get("model"),
        "action_required": action,
        "is_urgent": _normalize_urgent(order, instruction),
        "description": order.get("description", order.get("remark")),
    }


def _validation_error(sample: dict) -> str | None:
    """返回样本校验失败原因；通过时返回 None。"""
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
    """验证单条样本是否合法。"""
    return _validation_error(sample) is None


def _valid_samples(raw: list[dict]) -> list[dict]:
    """过滤并规范化样本。"""
    valid = []
    for sample in raw:
        normalized = _normalize_sample(sample)
        if validate_sample(normalized):
            valid.append(normalized)
    return valid


def _print_rejection_summary(raw: list[dict]) -> None:
    """输出一批样本全部失败时的诊断信息。"""
    if not raw:
        print("  DeepSeek 返回 samples 为空")
        return

    reasons = Counter(_validation_error(sample) or "valid" for sample in raw)
    reason_text = ", ".join(f"{reason}: {count}" for reason, count in reasons.items())
    preview = json.dumps(raw[0], ensure_ascii=False)[:500]
    print(f"  本批样本未通过校验：{reason_text}")
    print(f"  返回样例：{preview}")


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
        f"你是港口备件仓储数据标注专家。\n"
        f"知识库摘要（参考真实备件名称和型号）：\n{kb_context}\n\n"
        f"生成 {n} 条【{scenario}】场景训练数据，要求：{desc}\n\n"
        f"返回严格 JSON，外层只能是 samples：\n"
        f'{{"samples":[{{"input":"用户指令","output":"'
        f'[{{\\"part_name\\":\\"备件名\\",\\"quantity\\":1,'
        f'\\"model\\":\\"型号或null\\",\\"action_required\\":\\"出库\\",'
        f'\\"is_urgent\\":false,\\"description\\":null}}]"}}, ...]}}\n\n'
        f"output 必须是 WorkOrder JSON 数组字符串，每个对象只能包含 "
        f"part_name、quantity、model、action_required、is_urgent、description 六个字段；"
        f"不要使用 name/type/operation/priority/location/unit 等字段。"
        f"action_required 只能是 入库/出库/调库/null；is_urgent 必须是 true 或 false；"
        f"无法确定的字段输出 null，指令表达多样化。"
    )
    resp = _get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        raw = data.get("samples", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"  DeepSeek 返回不是预期 JSON：{str(content)[:500]}")
        return []
    valid = _valid_samples(raw)
    if not valid:
        _print_rejection_summary(raw)
    return valid


def deduplicate(
    samples: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """按 input 文本去重。"""
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
