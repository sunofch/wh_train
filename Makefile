# ─────────────────────────────────────────────────────────
#  wh_train Makefile — 统一命令入口
#
#  变量（可通过命令行覆盖）：
#    GPU        使用哪块 GPU（默认 1）
#    BASE_MODEL 基座模型（默认 Qwen/Qwen3.5-4B）
#    ADAPTER    LoRA adapter 目录（默认 output/qwen35_lora_vllm）
#    KB_DIR     知识库目录（或设置环境变量 WH_KB_DIR）
#    DATA_DIR   数据目录（默认 data）
# ─────────────────────────────────────────────────────────

GPU        ?= 1
BASE_MODEL ?= Qwen/Qwen3.5-4B
ADAPTER    ?= output/qwen35_lora_vllm
KB_DIR     ?= $(WH_KB_DIR)
DATA_DIR   ?= data

TRAIN_CFG  := config/train_config_vllm_lora.yaml
PRED_DIR   := $(ADAPTER)/predict

.PHONY: help data train eval chat serve check check-data post-train test clean

help:
	@echo "用法：make <target> [变量=值]"
	@echo ""
	@echo "  data        生成训练数据（需 DEEPSEEK_API_KEY）"
	@echo "  train       LoRA 微调训练"
	@echo "  eval        批量预测 + 评估指标"
	@echo "  chat        交互式对话（llamafactory-cli）"
	@echo "  serve       启动 vLLM 服务"
	@echo "  check       vLLM 服务健康检查"
	@echo "  check-data  数据质量检查"
	@echo "  post-train  生成训练元数据摘要（run_info.json）"
	@echo "  test        运行单元测试"
	@echo "  clean       清理 __pycache__"
	@echo ""
	@echo "示例："
	@echo "  make train GPU=0"
	@echo "  make eval  ADAPTER=output/qwen35_lora_vllm"
	@echo "  make data  KB_DIR=/path/to/knowledge_base"

# ── 数据生成 ──────────────────────────────────────────────
data:
	@test -n "$(DEEPSEEK_API_KEY)" || \
		(echo "错误：请先设置 DEEPSEEK_API_KEY"; exit 1)
	python generate_dataset.py \
		$(if $(KB_DIR),--kb-dir $(KB_DIR),) \
		--output-dir $(DATA_DIR)

# ── 训练 ─────────────────────────────────────────────────
train:
	HF_ENDPOINT=https://hf-mirror.com \
	CUDA_VISIBLE_DEVICES=$(GPU) \
		llamafactory-cli train $(TRAIN_CFG)

# ── 批量预测 + 评估 ───────────────────────────────────────
eval:
	CUDA_VISIBLE_DEVICES=$(GPU) llamafactory-cli train $(TRAIN_CFG) \
		do_train=false \
		do_predict=true \
		predict_with_generate=true \
		output_dir=$(PRED_DIR) \
		max_new_tokens=256 \
		temperature=0
	python evaluate.py \
		$(PRED_DIR)/generated_predictions.jsonl \
		$(DATA_DIR)/val.jsonl \
		--report $(PRED_DIR)/eval_report.json \
		--errors $(PRED_DIR)/eval_errors.jsonl

# ── 交互测试 ──────────────────────────────────────────────
chat:
	CUDA_VISIBLE_DEVICES=$(GPU) llamafactory-cli chat $(TRAIN_CFG) \
		adapter_name_or_path=$(ADAPTER) \
		finetuning_type=lora \
		do_train=false \
		max_new_tokens=256 \
		temperature=0

# ── vLLM 服务部署 ─────────────────────────────────────────
serve:
	CUDA_VISIBLE_DEVICES=$(GPU) vllm serve $(BASE_MODEL) \
		--trust-remote-code \
		--enable-lora \
		--lora-modules wh-qwen35=$(ADAPTER) \
		--max-lora-rank 16 \
		--served-model-name $(BASE_MODEL) \
		--host 0.0.0.0 \
		--port 8000 \
		--max-model-len 512 \
		--max-num-seqs 16 \
		--gpu-memory-utilization 0.80

# ── 健康检查 ──────────────────────────────────────────────
check:
	python scripts/healthcheck.py

# ── 数据质量报告 ──────────────────────────────────────────
check-data:
	python scripts/check_data.py --data-dir $(DATA_DIR)

# ── 训练后元数据汇总 ──────────────────────────────────────
post-train:
	python scripts/post_train.py --output-dir $(ADAPTER)

# ── 单元测试 ──────────────────────────────────────────────
test:
	pytest tests/ -v

# ── 清理 ─────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
