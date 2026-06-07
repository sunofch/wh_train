"""vLLM 服务健康检查脚本。

用法：
    python scripts/healthcheck.py [--host localhost] [--port 8000] [--model wh-qwen35]

检查内容：
  1. /health 端点连通性
  2. /v1/models 可见模型列表
  3. 发送测试推理请求，验证响应格式并报告延迟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TEST_INPUT = "出库2个轴承6208，紧急"
SYSTEM_PROMPT = (
    "你是港口备件指令解析助手。"
    "将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。"
    "action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"
)


def _get(url: str, timeout: int = 10) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  [错误] GET {url}：{e}")
        return None


def _post(url: str, payload: dict, timeout: int = 30) -> tuple[dict | None, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            return json.loads(resp.read().decode()), elapsed
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        elapsed = time.perf_counter() - t0
        print(f"  [错误] POST {url}：{e}")
        return None, elapsed


def check_health(base_url: str) -> bool:
    print(f"[1] 连通性检查  {base_url}/health")
    data = _get(f"{base_url}/health")
    if data is not None:
        print(f"  OK  响应：{data}")
        return True
    # 部分 vLLM 版本没有 /health，尝试 /v1/models
    data = _get(f"{base_url}/v1/models")
    if data is not None:
        print(f"  OK（via /v1/models）")
        return True
    print("  FAIL：服务不可达")
    return False


def check_models(base_url: str, model_name: str) -> bool:
    print(f"\n[2] 模型列表检查  {base_url}/v1/models")
    data = _get(f"{base_url}/v1/models")
    if data is None:
        return False
    model_ids = [m.get("id", "") for m in data.get("data", [])]
    print(f"  已加载模型：{model_ids}")
    if model_name in model_ids:
        print(f"  OK：{model_name} 已加载")
        return True
    print(f"  WARN：未发现 {model_name}，请检查 --lora-modules 参数")
    return False


def check_inference(base_url: str, model_name: str) -> bool:
    print(f"\n[3] 推理验证  model={model_name}")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": TEST_INPUT},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    data, elapsed = _post(f"{base_url}/v1/chat/completions", payload)
    if data is None:
        return False

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"  延迟：{elapsed:.2f}s")
    print(f"  输入：{TEST_INPUT!r}")
    print(f"  输出：{content!r}")

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed:
            print(f"  OK：JSON 解析成功，共 {len(parsed)} 个 WorkOrder")
            return True
        print("  WARN：输出不是非空 JSON 数组")
    except json.JSONDecodeError:
        print("  WARN：输出无法解析为 JSON")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 服务健康检查")
    parser.add_argument("--host",  default="localhost")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--model", default="wh-qwen35", help="LoRA 模型名称")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"目标服务：{base_url}  model={args.model}\n")

    ok = True
    ok &= check_health(base_url)
    if ok:
        check_models(base_url, args.model)
        ok &= check_inference(base_url, args.model)

    print(f"\n{'✓ 健康检查通过' if ok else '✗ 健康检查未完全通过'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
