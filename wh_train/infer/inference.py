"""离线推理：单条 / 批量 / 交互。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wh_train.schema import SYSTEM_PROMPT


def build_messages(user_text: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]


def extract_user_text(record: dict) -> str | None:
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return record.get("input")


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def load_model(base_model: str, adapter_path: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载 tokenizer：{base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    print(f"加载模型：{base_model}  device={device}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True)
    if adapter_path:
        from peft import PeftModel
        print(f"挂载 LoRA adapter：{adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, user_text: str, max_new_tokens: int,
                system_prompt: str = SYSTEM_PROMPT) -> str:
    import torch
    prompt = _apply_chat_template(tokenizer, build_messages(user_text, system_prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_batch(model, tokenizer, input_path: str, output_path: str, max_new_tokens: int,
              system_prompt: str = SYSTEM_PROMPT) -> None:
    lines = [l for l in Path(input_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(lines)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, line in enumerate(lines, 1):
            record = json.loads(line)
            user_text = extract_user_text(record)
            if user_text is None:
                print(f"  [警告] 第 {i} 条找不到用户输入，跳过", file=sys.stderr)
                continue
            prediction = predict_one(model, tokenizer, user_text, max_new_tokens, system_prompt)
            gold = ""
            if "messages" in record:
                gold = record["messages"][-1].get("content", "")
            elif "label" in record:
                gold = record["label"]
            elif "output" in record:
                gold = record["output"]
            out_record: dict = {"predict": prediction}
            if gold:
                out_record["label"] = gold
            if record.get("scenario"):
                out_record["scenario"] = record["scenario"]
            out_record["input"] = user_text
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            print(f"  [{i}/{total}] {user_text[:40]!r} → {prediction[:60]!r}", file=sys.stderr)
    print(f"批量推理完成，结果写入：{output_path}", file=sys.stderr)


def run_interactive(model, tokenizer, max_new_tokens: int,
                    system_prompt: str = SYSTEM_PROMPT) -> None:
    print("进入交互模式（Ctrl+C / Ctrl+D 退出）", file=sys.stderr)
    try:
        while True:
            try:
                user_text = input(">>> ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            print(predict_one(model, tokenizer, user_text, max_new_tokens, system_prompt))
    except KeyboardInterrupt:
        pass
