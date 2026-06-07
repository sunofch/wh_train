"""[转发壳] 实现已迁移至 wh_train.infer.inference。"""
from __future__ import annotations

import argparse

from wh_train.infer.inference import (  # noqa: F401
    build_messages, extract_user_text, load_model,
    predict_one, run_batch, run_interactive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线推理脚本（单条 / 批量 / 交互）")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--text", default=None)
    parser.add_argument("--input", default=None, metavar="PATH")
    parser.add_argument("--output", default=None, metavar="PATH")
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
