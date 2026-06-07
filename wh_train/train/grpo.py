"""GRPO 强化精修训练（trl GRPOTrainer + vLLM rollout）。"""
from __future__ import annotations

import inspect

from wh_train.schema import SYSTEM_PROMPT
from wh_train.utils.io import read_jsonl


def build_grpo_rows(train_jsonl: str) -> list[dict]:
    """从 messages 格式 jsonl 构建 GRPO 行：{prompt: 消息列表, gold: 标准答案串}。

    GRPO 不需要 assistant 目标作为标签，assistant 内容作为 gold 供奖励函数判分。
    跳过缺少 user 消息的记录。
    """
    rows = []
    for rec in read_jsonl(train_jsonl):
        msgs = rec.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), None)
        assistant = next((m for m in msgs if m.get("role") == "assistant"), None)
        if user is None or assistant is None:
            continue
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user["content"]},
            ],
            "gold": assistant["content"],
        })
    return rows


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    """Keep only kwargs accepted by the installed TRL version."""
    params = inspect.signature(callable_obj).parameters
    supported = {k: v for k, v in kwargs.items() if k in params}
    ignored = sorted(set(kwargs) - set(supported))
    if ignored:
        print(f"提示：当前 TRL 不支持这些 GRPOConfig 参数，已忽略：{', '.join(ignored)}")
    return supported


def run_grpo(config_path: str) -> None:
    """按配置执行 GRPO 训练。

    双卡用法（先在 GPU1 起 vLLM server，再在 GPU0 起训练）：
      # 终端 1（GPU1，rollout 生成）
      CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
          --model Qwen/Qwen3.5-4B --port 8000
      # 终端 2（GPU0，策略训练）
      CUDA_VISIBLE_DEVICES=0 python -m wh_train grpo --config config/grpo.yaml
    """
    import trl.import_utils as trl_import_utils
    # trl==0.24.0 may return a truthy (False, None) tuple for vllm_ascend
    # availability, causing CUDA runs to import vllm_ascend incorrectly.
    trl_import_utils._vllm_ascend_available = False

    from datasets import Dataset
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText
    from trl import GRPOConfig, GRPOTrainer

    from wh_train.config import load_yaml
    from wh_train.reward.reward_fn import reward_func

    cfg = load_yaml(config_path)
    rows = build_grpo_rows(cfg["train_jsonl"])
    if not rows:
        raise RuntimeError(f"未从 {cfg['train_jsonl']} 构建出任何 GRPO 样本")
    dataset = Dataset.from_list(rows)

    align = cfg.get("align", "positional")

    def _reward(completions, gold, **kwargs):
        # trl 传入 completions（list[str]）与数据集透传列 gold；其余列在 kwargs
        return reward_func(completions=completions, gold=gold, align=align)

    sft_adapter = cfg.get("sft_adapter")
    if sft_adapter:
        print(f"提示：从 SFT adapter 继续 GRPO：{sft_adapter}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            cfg["base_model"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, sft_adapter, is_trainable=True)
        peft_config = None
    else:
        model = cfg["base_model"]
        peft_config = LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_target"].split(","),
            task_type="CAUSAL_LM",
        )

    grpo_kwargs = {
        "output_dir": cfg["output_dir"],
        "num_generations": cfg["num_generations"],
        "temperature": cfg["temperature"],
        "max_prompt_length": cfg.get("max_prompt_length"),
        "max_completion_length": cfg["max_completion_length"],
        "beta": cfg["beta"],
        "learning_rate": cfg["learning_rate"],
        "num_train_epochs": cfg["num_train_epochs"],
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "logging_steps": cfg["logging_steps"],
        "save_steps": cfg["save_steps"],
        "bf16": True,
        "use_vllm": cfg.get("use_vllm", True),
        "vllm_mode": cfg.get("vllm_mode", "server"),
        "vllm_server_host": cfg.get("vllm_server_host", "127.0.0.1"),
        "vllm_server_port": cfg.get("vllm_server_port", 8000),
        "vllm_group_port": cfg.get("vllm_group_port", 51216),
        "report_to": "tensorboard",
    }
    grpo_config = GRPOConfig(**_filter_supported_kwargs(GRPOConfig.__init__, grpo_kwargs))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_reward,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"GRPO 完成，adapter 保存至：{cfg['output_dir']}")
