"""训练后信息汇总脚本，将关键元数据写入 run_info.json。

用法：
    python scripts/post_train.py --output-dir output/qwen35_lora_vllm

写出内容（output_dir/run_info.json）：
  - run_date：生成时间
  - output_dir：训练输出目录
  - base_model：基座模型（来自 adapter_config.json）
  - adapter_config：完整 LoRA 配置
  - best_checkpoint：验证集 loss 最低的 checkpoint 路径
  - best_eval_loss：最优 eval loss
  - final_step：最终训练步数
  - total_epochs：训练轮数
  - train_results：train_results.json 内容
  - eval_results：eval_results.json 内容
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _find_best_checkpoint(trainer_state: dict, output_dir: Path) -> tuple[str | None, float | None]:
    """从 trainer_state 中找 eval_loss 最低的 checkpoint。"""
    best_path = trainer_state.get("best_model_checkpoint")
    if best_path:
        return best_path, trainer_state.get("best_metric")

    # 若 load_best_model_at_end 未生效，从 log_history 手动找
    log_history = trainer_state.get("log_history", [])
    best_loss: float | None = None
    best_step: int | None = None
    for entry in log_history:
        if "eval_loss" in entry:
            loss = entry["eval_loss"]
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_step = entry.get("step")

    if best_step is not None:
        candidate = output_dir / f"checkpoint-{best_step}"
        return str(candidate) if candidate.exists() else None, best_loss

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="训练后信息汇总脚本")
    parser.add_argument(
        "--output-dir", required=True,
        help="训练输出目录（如 output/qwen35_lora_vllm）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"错误：目录不存在：{output_dir}")
        raise SystemExit(1)

    trainer_state  = _read_json(output_dir / "trainer_state.json")
    adapter_config = _read_json(output_dir / "adapter_config.json")
    train_results  = _read_json(output_dir / "train_results.json")
    eval_results   = _read_json(output_dir / "eval_results.json")
    all_results    = _read_json(output_dir / "all_results.json")

    best_ckpt, best_loss = _find_best_checkpoint(trainer_state, output_dir)

    run_info = {
        "run_date":        datetime.now(timezone.utc).isoformat(),
        "output_dir":      str(output_dir.resolve()),
        "base_model":      adapter_config.get("base_model_name_or_path", "unknown"),
        "adapter_config":  adapter_config,
        "best_checkpoint": best_ckpt,
        "best_eval_loss":  best_loss,
        "final_step":      trainer_state.get("global_step"),
        "total_epochs":    trainer_state.get("epoch"),
        "train_results":   train_results or {k: v for k, v in all_results.items() if "train" in k},
        "eval_results":    eval_results  or {k: v for k, v in all_results.items() if "eval" in k},
    }

    out_path = output_dir / "run_info.json"
    out_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_info.json 已写入：{out_path}")

    # 打印摘要
    print(f"\n── 训练摘要 ──")
    print(f"  基座模型：{run_info['base_model']}")
    print(f"  输出目录：{run_info['output_dir']}")
    print(f"  最优 checkpoint：{run_info['best_checkpoint'] or '（未记录，请检查 load_best_model_at_end 配置）'}")
    print(f"  最优 eval_loss：{run_info['best_eval_loss']}")
    print(f"  最终步数：{run_info['final_step']}  轮数：{run_info['total_epochs']}")
    if train_results:
        print(f"  训练 loss：{train_results.get('train_loss', 'N/A'):.4f}")
    if eval_results:
        print(f"  验证 loss：{eval_results.get('eval_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
