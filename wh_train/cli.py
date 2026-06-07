"""统一 CLI 入口：python -m wh_train <command>。

子命令只负责解析参数 → 调对应模块，逻辑全在模块内。
"""
from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wh_train", description="港口备件指令解析微调工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-data", help="生成训练数据（需 DEEPSEEK_API_KEY）")
    p_gen.add_argument("--kb-dir", default=os.getenv("WH_KB_DIR", ""))
    p_gen.add_argument("--output-dir", default="data")

    p_sft = sub.add_parser("sft", help="SFT 训练（调 LLaMA-Factory）")
    p_sft.add_argument("--config", required=True)

    p_grpo = sub.add_parser("grpo", help="GRPO 强化精修")
    p_grpo.add_argument("--config", required=True)

    p_eval = sub.add_parser("eval", help="评估")
    p_eval.add_argument("--pred", required=True)
    p_eval.add_argument("--gold", required=True)
    p_eval.add_argument("--report", default=None)
    p_eval.add_argument("--errors", default=None)

    p_inf = sub.add_parser("infer", help="推理（单条/批量/交互）")
    p_inf.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p_inf.add_argument("--adapter", default=None)
    p_inf.add_argument("--gpu", type=int, default=0)
    p_inf.add_argument("--max-tokens", type=int, default=256)
    p_inf.add_argument("--text", default=None)
    p_inf.add_argument("--input", default=None)
    p_inf.add_argument("--output", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "gen-data":
        from wh_train.data.generate import run
        run(args.kb_dir, args.output_dir)

    elif args.command == "sft":
        from wh_train.train.sft import run_sft
        run_sft(args.config)

    elif args.command == "grpo":
        from wh_train.train.grpo import run_grpo
        run_grpo(args.config)

    elif args.command == "eval":
        from wh_train.eval.evaluate import evaluate_file, print_results
        metrics = evaluate_file(args.pred, args.gold, report_path=args.report, errors_path=args.errors)
        print_results(metrics)

    elif args.command == "infer":
        from wh_train.infer.inference import (
            load_model, predict_one, run_batch, run_interactive,
        )
        device = f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"
        model, tokenizer = load_model(args.base_model, args.adapter, device)
        if args.text:
            print(predict_one(model, tokenizer, args.text, args.max_tokens))
        elif args.input:
            out_path = args.output or args.input.replace(".jsonl", "_pred.jsonl")
            run_batch(model, tokenizer, args.input, out_path, args.max_tokens)
        else:
            run_interactive(model, tokenizer, args.max_tokens)
