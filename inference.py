"""离线推理脚本，支持单条文本和批量 JSONL 文件两种模式。

用法：
    # 单条文本
    python inference.py --adapter output/qwen35_lora_vllm --text "出库2个轴承6208"

    # 批量推理（读取 JSONL，写出预测）
    python inference.py --adapter output/qwen35_lora_vllm \\
        --input data/val.jsonl --output output/pred.jsonl

    # 仅使用基座模型（不挂载 adapter）
    python inference.py --base-model Qwen/Qwen3.5-4B --text "出库2个轴承6208"

输出 JSONL 格式与 LLaMA-Factory generated_predictions.jsonl 兼容（含 predict 字段），
可直接传入 evaluate.py 计算指标。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)


def _build_messages(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """应用聊天模板，兼容 Qwen3 的 enable_thinking 参数。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _extract_user_text(record: dict) -> str | None:
    """从 JSONL 记录中提取用户输入文本。"""
    if "messages" in record:
        for msg in record["messages"]:
            if msg.get("role") == "user":
                return msg["content"]
    return record.get("input")


def load_model(base_model: str, adapter_path: str | None, device: str):
    """加载模型和 tokenizer，可选挂载 LoRA adapter。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载 tokenizer：{base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )

    print(f"加载模型：{base_model}  device={device}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel
        print(f"挂载 LoRA adapter：{adapter_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def predict_one(model, tokenizer, user_text: str, max_new_tokens: int) -> str:
    """对单条用户输入做推理，返回助手输出字符串。"""
    import torch

    messages = _build_messages(user_text)
    prompt = _apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_batch(
    model,
    tokenizer,
    input_path: str,
    output_path: str,
    max_new_tokens: int,
) -> None:
    """批量推理：读取 input_path JSONL，写出 output_path JSONL。"""
    lines = [l for l in Path(input_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(lines)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, line in enumerate(lines, 1):
            record = json.loads(line)
            user_text = _extract_user_text(record)
            if user_text is None:
                print(f"  [警告] 第 {i} 条记录找不到用户输入，已跳过", file=sys.stderr)
                continue

            prediction = predict_one(model, tokenizer, user_text, max_new_tokens)

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
            print(f"  [{i}/{total}] {user_text[:40]!r}  →  {prediction[:60]!r}", file=sys.stderr)

    print(f"批量推理完成，结果已写入：{output_path}", file=sys.stderr)


def run_interactive(model, tokenizer, max_new_tokens: int) -> None:
    """交互推理模式，从 stdin 逐行读取，Ctrl+C 或 EOF 退出。"""
    print("进入交互模式（输入指令后回车，Ctrl+C 或 Ctrl+D 退出）", file=sys.stderr)
    try:
        while True:
            try:
                user_text = input(">>> ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            result = predict_one(model, tokenizer, user_text, max_new_tokens)
            print(result)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线推理脚本（单条 / 批量 / 交互）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.5-4B",
        help="基座模型路径或 HuggingFace model ID（默认 Qwen/Qwen3.5-4B）",
    )
    parser.add_argument(
        "--adapter", default=None,
        help="LoRA adapter 目录（output/qwen35_lora_vllm）",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="使用的 GPU 编号，-1 表示 CPU（默认 0）",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="最大生成 token 数（默认 256）",
    )
    parser.add_argument("--text", default=None, help="单条用户输入文本")
    parser.add_argument("--input",  default=None, metavar="PATH", help="批量推理输入 JSONL")
    parser.add_argument("--output", default=None, metavar="PATH", help="批量推理输出 JSONL")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
    model, tokenizer = load_model(args.base_model, args.adapter, device)

    if args.text:
        print(predict_one(model, tokenizer, args.text, args.max_tokens))
    elif args.input:
        out_path = args.output or args.input.replace(".jsonl", "_pred.jsonl")
        run_batch(model, tokenizer, args.input, out_path, args.max_tokens)
    else:
        run_interactive(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
