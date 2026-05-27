# wh_train - 港口备件指令解析微调

基于 Qwen3.5-4B + LLaMA-Factory LoRA，对港口备件领料指令进行微调，将自然语言指令解析为结构化 WorkOrder JSON 数组。

训练完成后保留 LoRA adapter，由 vLLM 在基座模型上直接挂载 adapter，替换 `wh_graphrag_re` 中的 ReAct 多轮 Agent。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU  | NVIDIA RTX 3090 Ti 24 G（或同等显存） |
| CUDA | 12.2 |
| Python | 3.11 |

## 安装

```bash
# 1. 创建 conda 环境
conda create -n train python=3.11 -y
conda activate train

# 2. 安装 PyTorch（CUDA 12.4 wheel 兼容 CUDA 12.2 驱动）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3. 安装其余依赖
pip install -r requirements.txt

# 4. 安装 LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
llamafactory-cli version
cd /home/catlab/wh_train
```

---

## 项目结构

```
wh_train/
├── config/
│   ├── train_config.yaml    # LLaMA-Factory LoRA 训练配置
│   ├── train_config_vllm_lora.yaml # vLLM LoRA 热挂载兼容训练配置
│   └── export_config.yaml   # 可选 LoRA 合并导出配置
├── data/
│   ├── train.jsonl          # 训练集（ShareGPT 格式）
│   ├── val.jsonl            # 验证集
│   └── dataset_info.json    # LLaMA-Factory 数据集注册
├── tests/
│   ├── test_generate_utils.py
│   └── test_generate_main.py
├── generate_dataset.py      # DeepSeek API 合成数据生成
├── evaluate.py              # 验证集评估（JSON 解析率 / 字段准确率等）
├── requirements.txt
└── README.md
```

---

## 使用步骤

### 1. 生成训练数据

需要配置 DeepSeek API Key：

```bash
export DEEPSEEK_API_KEY="your-key-here"
python generate_dataset.py
```

生成完成后输出 `data/train.jsonl`（约 1350 条）和 `data/val.jsonl`（约 150 条）。

### 2. 训练

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train config/train_config_vllm_lora.yaml
```

`config/train_config_vllm_lora.yaml` 只训练 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，避免 vLLM 热挂载 Qwen3.5 LoRA 时在 fused/linear-attention 模块上触发兼容性问题。

训练日志与 checkpoint 保存至 `output/qwen35_lora_vllm/`。`config/train_config.yaml` 使用 `lora_target: all`，适合导出合并模型，不建议直接作为 vLLM LoRA adapter 热挂载。

### 3. 交互试用

优先使用验证集 loss 最低的 checkpoint，例如：

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli chat \
  config/train_config.yaml \
  adapter_name_or_path=output/qwen35_lora_vllm \
  finetuning_type=lora \
  do_train=false \
  max_new_tokens=256 \
  temperature=0
```

进入交互模式后输入自然语言指令，例如：

```text
紧急出库2个6208-2RS-C3-SKF轴承和1个HTW-3400Nm-ENERPAC液压力矩扳手
```

### 4. 批量评估

先用 LLaMA-Factory 对验证集生成预测：

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
  config/train_config.yaml \
  do_train=false \
  do_predict=true \
  predict_with_generate=true \
  output_dir=output/qwen35_lora_vllm/predict \
  max_new_tokens=256 \
  temperature=0
```

再计算指标：

```bash
python evaluate.py output/qwen35_lora_vllm/predict/generated_predictions.jsonl data/val.jsonl
```

输出指标：

| 指标 | 目标 |
|------|------|
| JSON 解析成功率 | > 98% |
| 字段准确率（各字段平均） | > 90% |
| 数组长度准确率（多备件） | > 85% |
| null 召回率（缺失字段） | > 95% |

### 5. vLLM 使用 LoRA adapter 部署

训练输出目录 `output/qwen35_lora_vllm/` 本身就是 PEFT LoRA adapter，包含 `adapter_config.json` 和 `adapter_model.safetensors`。vLLM 可直接加载基座模型并挂载该 adapter，无需合并权重。

建议使用单独的 serving 环境安装较新的 vLLM。`vllm==0.18.0` 在 Qwen3.5 LoRA 热挂载上仍可能触发启动阶段兼容问题，优先使用已包含 Qwen3.5 LoRA 修复的新版本。

RTX 3090 Ti 24G 上不要使用默认 256 dummy requests warmup；显存容易在 sampler warmup 阶段 OOM。降低 `max_num_seqs` 和 `gpu_memory_utilization`：

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-4B \
  --trust-remote-code \
  --enable-lora \
  --lora-modules wh-qwen35=./output/qwen35_lora_vllm \
  --max-lora-rank 16 \
  --served-model-name Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 512 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80
```

如果仍然 OOM，再降到 `--max-num-seqs 8 --gpu-memory-utilization 0.70`，或加 `--enforce-eager` 绕过 CUDA graph warmup。

调用 OpenAI 兼容接口时，`model` 使用 LoRA 名称 `wh-qwen35`：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "wh-qwen35",
    "messages": [
      {"role": "system", "content": "你是港口备件指令解析助手。将用户指令解析为JSON数组，无法确定的字段输出null，不要猜测。action_required 只能是 入库/出库/调库/null，is_urgent 为 bool。"},
      {"role": "user", "content": "紧急出库2个6208-2RS-C3-SKF轴承"}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

如需导出合并后的完整模型，仍可执行：

```bash
llamafactory-cli export config/export_config.yaml
```

---

## 修改训练参数

编辑 `config/train_config.yaml`，关键参数说明：

```yaml
lora_rank: 16
learning_rate: 0.0002
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
```

---

## 集成到 wh_graphrag_re

训练完成后，修改 `/home/catlab/wh/wh_graphrag_re/.env`：

```ini
VLM35_BASE_MODEL=Qwen/Qwen3.5-4B
VLM35_ADAPTER=./output/qwen35_lora_vllm
VLM35_MODEL=wh-qwen35
```

详见设计文档：`docs/superpowers/specs/2026-05-25-instruction-finetune-design.md`

---

## 运行测试

```bash
conda activate train
cd /home/catlab/wh_train
pytest tests/ -v
```

---

## 数据场景说明

见 `data.md`，包含 7 类场景（显式标准指令 / 隐式行为推断 / 隐式紧急推断 / 字段缺失 / 多备件 / ASR 噪声 / 数量模糊）及输入输出示例。
