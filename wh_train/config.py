"""配置加载：yaml → dict。供 CLI 子命令读取训练/数据配置。"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    """读取 yaml 配置为 dict；文件不存在抛 FileNotFoundError。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在：{p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
