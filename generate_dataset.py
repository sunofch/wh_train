"""[转发壳] 实现已迁移至 wh_train.data.*。保留顶层导入兼容。"""
from __future__ import annotations

import argparse
import os

from wh_train.schema import SYSTEM_PROMPT  # noqa: F401  (兼容旧引用)
from wh_train.data.scenarios import SCENARIOS  # noqa: F401
from wh_train.data.normalize import validate_sample  # noqa: F401
from wh_train.data.split import (  # noqa: F401
    to_openai_messages_format, load_knowledge_base, stratified_split, deduplicate,
)
from wh_train.data import generate as _gen
from wh_train.data.generate import run  # noqa: F401


def _get_client():
    """委托包内懒初始化客户端；保留本名以兼容既有 mock 路径。"""
    return _gen._get_client()


def generate_batch(scenario: str, desc: str, kb_context: str, n: int = 10) -> list[dict]:
    """转发到包内实现，注入本模块的 _get_client（便于测试 patch）。"""
    return _gen.generate_batch(scenario, desc, kb_context, n, client=_get_client())


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek API 合成训练数据生成脚本")
    parser.add_argument("--kb-dir", default=os.getenv("WH_KB_DIR", ""))
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    run(args.kb_dir, args.output_dir)


if __name__ == "__main__":
    main()
