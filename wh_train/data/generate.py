"""DeepSeek API 合成训练数据生成主流程。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from wh_train.schema import SYSTEM_PROMPT
from wh_train.data.scenarios import SCENARIOS
from wh_train.data.normalize import valid_samples, rejection_summary
from wh_train.data.split import (
    to_openai_messages_format, load_knowledge_base, stratified_split, deduplicate,
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", "placeholder"),
            base_url="https://api.deepseek.com",
        )
    return _client


def generate_batch(scenario: str, desc: str, kb_context: str, n: int = 10, client=None) -> list[dict]:
    """调用 DeepSeek 生成一批样本，过滤无效后返回。"""
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
    cli = client or _get_client()
    resp = cli.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        raw = json.loads(content).get("samples", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"  DeepSeek 返回不是预期 JSON：{str(content)[:500]}")
        return []
    valid = valid_samples(raw)
    if not valid:
        print(f"  本批样本未通过校验：{rejection_summary(raw)}")
    return valid


def run(kb_dir: str, output_dir: str) -> tuple[int, int]:
    """生成全部场景数据并写出 train/val.jsonl，返回 (train_n, val_n)。"""
    kb_context = load_knowledge_base(kb_dir) if kb_dir else ""
    if not kb_context:
        print("警告：未加载到知识库内容，生成数据将不含真实备件参考。")
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    all_samples: list[tuple[str, dict]] = []
    for scenario, cfg in SCENARIOS.items():
        print(f"\n生成 {scenario}（目标 {cfg['count']} 条）...")
        collected: list[tuple[str, dict]] = []
        batch_size, max_iter = 10, cfg["count"] * 3
        for _ in range(0, max_iter, batch_size):
            if len(collected) >= cfg["count"]:
                break
            batch = generate_batch(scenario, cfg["desc"], kb_context, batch_size)
            collected.extend((scenario, s) for s in batch)
            print(f"  已收集 {len(collected)} / {cfg['count']}")
        all_samples.extend(collected[: cfg["count"]])

    all_samples = deduplicate(all_samples)
    train_raw, val_raw = stratified_split(all_samples, val_ratio=0.1, seed=42)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_raw:
            f.write(json.dumps(to_openai_messages_format(s, SYSTEM_PROMPT), ensure_ascii=False) + "\n")
    with open(out / "val.jsonl", "w", encoding="utf-8") as f:
        for s in val_raw:
            f.write(json.dumps(to_openai_messages_format(s, SYSTEM_PROMPT), ensure_ascii=False) + "\n")
    print(f"\n完成：train {len(train_raw)} 条，val {len(val_raw)} 条")
    return len(train_raw), len(val_raw)
