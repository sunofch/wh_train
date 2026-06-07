"""SFT 训练薄封装：委托 LLaMA-Factory CLI。"""
from __future__ import annotations

import subprocess


def run_sft(config_path: str) -> int:
    """调用 llamafactory-cli train 执行 SFT，返回退出码。"""
    cmd = ["llamafactory-cli", "train", config_path]
    print(f"执行：{' '.join(cmd)}")
    return subprocess.call(cmd)
