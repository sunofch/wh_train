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
├── Makefile                 # 常用命令入口
├── README.md
├── requirements.txt
├── pytest.ini
├── config/
│   ├── train_config.yaml    # LLaMA-Factory LoRA 训练配置
│   ├── train_config_vllm_lora.yaml # vLLM LoRA 热挂载兼容训练配置
│   ├── chat_config.yaml     # LLaMA-Factory chat / inference 配置
│   ├── grpo.yaml            # TRL GRPO 强化精修配置
│   └── export_config.yaml   # 可选 LoRA 合并导出配置
├── wh_train/                # Python 包源码
│   ├── __main__.py          # python -m wh_train 入口
│   ├── cli.py               # 统一 CLI 入口
│   ├── schema.py            # WorkOrder 字段、合法枚举、系统提示词
│   ├── data/                # 数据生成、归一化、划分
│   ├── eval/                # 评估主流程
│   ├── infer/               # 离线推理
│   ├── reward/              # 解析、指标、GRPO 奖励函数
│   └── train/               # SFT / GRPO 训练入口
├── data/
│   ├── dataset_info.json    # LLaMA-Factory 数据集注册，建议保留
│   ├── train.jsonl          # 生成的训练集，忽略入库
│   └── val.jsonl            # 生成的验证集，忽略入库
├── scripts/
│   ├── check_data.py        # 数据质量检查
│   ├── healthcheck.py       # vLLM 服务健康检查
│   └── post_train.py        # 训练后元数据摘要
├── tests/                   # 单元测试
└── output/                  # 训练产物，忽略入库
```

根目录不再保留 `generate_dataset.py`、`evaluate.py`、`inference.py` 这类转发壳。统一使用 `python -m wh_train <command>` 或 `make <target>`。

---

## 使用步骤

### 1. 查看命令

```bash
python -m wh_train --help
make help
```

主要子命令：

| 命令 | 用途 |
|------|------|
| `python -m wh_train gen-data` | 生成训练/验证数据 |
| `python -m wh_train sft --config ...` | 调用 LLaMA-Factory 做 SFT |
| `python -m wh_train grpo --config ...` | TRL GRPO 强化精修 |
| `python -m wh_train eval --pred ... --gold ...` | 批量评估 |
| `python -m wh_train infer` | 单条、批量或交互式离线推理 |

### 2. 生成训练数据

需要配置 DeepSeek API Key。默认输出到 `data/`：

```bash
export DEEPSEEK_API_KEY="your-key-here"
python -m wh_train gen-data
```

生成完成后输出 `data/train.jsonl`（约 1350 条）和 `data/val.jsonl`（约 150 条）。

也可以通过 Makefile：

```bash
make data
```

### 3. 训练

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train config/train_config_vllm_lora.yaml
```

`config/train_config_vllm_lora.yaml` 只训练 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，避免 vLLM 热挂载 Qwen3.5 LoRA 时在 fused/linear-attention 模块上触发兼容性问题。

训练日志与 checkpoint 保存至 `output/qwen35_lora_vllm/`。`config/train_config.yaml` 使用 `lora_target: all`，适合导出合并模型，不建议直接作为 vLLM LoRA adapter 热挂载。

也可以通过 Makefile：

```bash
make train GPU=1
```

### 4. 交互试用

优先使用验证集 loss 最低的 checkpoint，例如：

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli chat \
  config/chat_config.yaml \
  model_name_or_path=Qwen/Qwen3.5-4B \
  adapter_name_or_path=output/qwen35_lora_vllm \
  finetuning_type=lora \
  max_new_tokens=256 \
  temperature=0
```

进入交互模式后输入自然语言指令，例如：

```text
紧急出库2个6208-2RS-C3-SKF轴承和1个HTW-3400Nm-ENERPAC液压力矩扳手
```

也可以通过 Makefile：

```bash
make chat GPU=1
```

### 5. 批量评估

先用 LLaMA-Factory 对验证集生成预测：

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
  config/train_config.yaml \
  do_train=false \
  do_predict=true \
  predict_with_generate=true \
  adapter_name_or_path=output/qwen35_lora_vllm \
  finetuning_type=lora \
  output_dir=output/qwen35_lora_vllm/predict \
  max_new_tokens=256 \
  do_sample=false \
  temperature=1.0
```

再计算指标：

```bash
python -m wh_train eval \
  --pred output/qwen35_lora_vllm/predict/generated_predictions.jsonl \
  --gold data/val.jsonl
```

输出指标：

| 指标 | 目标 |
|------|------|
| JSON 解析成功率 | > 98% |
| 字段准确率（各字段平均） | > 90% |
| 数组长度准确率（多备件） | > 85% |
| null 召回率（缺失字段） | > 95% |

也可以通过 Makefile：

```bash
make eval GPU=1
```

测试基座模型（无 LoRA adapter）作为对照基线：

```bash
make base-eval GPU=1
```

结果写入 `output/base_model/predict/eval_report.json`。

### 基座 vs SFT 结果对比

基于 `data/val.jsonl`（143 条）的评估结果：

| 指标 | 基座（无 LoRA） | SFT LoRA | 提升 |
|------|--------------|----------|------|
| JSON 解析成功率 | 77.6% | **100%** | +22.4pp |
| part_name 准确率 | 68.9% | **89.8%** | +20.9pp |
| model 准确率 | 70.7% | **97.6%** | +26.9pp |
| quantity 准确率 | 66.5% | **97.0%** | +30.5pp |
| action_required 准确率 | 77.3% | **97.0%** | +19.8pp |
| is_urgent 准确率 | 74.3% | **98.8%** | +24.6pp |
| 数组长度准确率 | 77.6% | **100%** | +22.4pp |
| model null 召回率 | 62.2% | **98.8%** | +36.6pp |
| quantity null 召回率 | 15.0% | **96.8%** | +81.8pp |
| 错误样本数 | 88 | 58 | -30 |

**按场景细分（SFT）**：

| 场景 | n | JSON 解析 | action 准确率 | is_urgent 准确率 |
|------|---|----------|-------------|----------------|
| 显式标准名称型号 | 14 | 100% | 100% | 100% |
| 多备件 | 22 | 100% | 100% | 97.8% |
| 数量模糊 | 13 | 100% | 100% | 100% |
| 动作不确定 | 9 | 100% | 100% | 100% |
| 字段缺失 | 12 | 100% | 100% | 100% |
| ASR 噪声 | 15 | 100% | 93.3% | 100% |
| 隐式紧急推断 | 15 | 100% | 93.3% | 93.3% |
| 显式非标准名称缺失型号 | 14 | 100% | 100% | 100% |
| 描述信息 | 10 | 100% | 90.0% | 100% |
| 隐式行为推断 | 19 | 100% | 89.5% | 100% |

**主要发现**：
- `quantity_null_recall` 提升最显著（+82pp）：基座模型对模糊数量几乎总是瞎猜，SFT 后学会输出 `null`
- `动作不确定` 场景基座模型 JSON 解析率为 0%，SFT 后达到 100%
- SFT 的 `part_name_accuracy`（89.8%）是各字段中最低的，主要原因是训练数据中 part_name 标注口径不统一（用户简称与标准全称混用、修饰词是否并入名称不一致）

### 6. vLLM 使用 LoRA adapter 部署

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

### 7. GRPO 强化精修

GRPO 使用 TRL + vLLM rollout，依赖组合和 SFT/eval 环境不同。不要直接在当前 `train` 环境里把 vLLM 升级到 GRPO 版本，否则会和 `transformers==5.6.0`、`torch==2.6.0+cu124` 冲突。

`requirements-grpo.txt` 当前锁定为 `trl==1.5.1`、`vllm==0.19.1`、`transformers==5.6.0`、`torch==2.10.0`。这里保留 `transformers==5.6.0` 是因为旧版 4.x Transformers 不能识别 Qwen3.5 的 `qwen3_5` 架构。

建议单独创建环境：

```bash
conda create -n wh-grpo python=3.11 -y
conda activate wh-grpo
pip install -r requirements-grpo.txt
```

GRPO 入口使用 `trl.GRPOTrainer`，不需要安装 LLaMA-Factory。不要在该环境里执行 `pip install -e /home/catlab/LLaMA-Factory`，否则 LLaMA-Factory 的依赖可能把 `trl`、`starlette` 等包改回不兼容版本。

进入项目目录：

```bash
cd /home/catlab/wh_train
```

先在一张卡启动 rollout 服务。优先使用本地 Hugging Face snapshot，并开启离线模式，避免 vLLM 启动时联网调用 `list_repo_files` 失败：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 trl vllm-serve \
  --model /home/catlab/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 512 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager
```

如果使用 `--model Qwen/Qwen3.5-4B` 直接从仓库名启动，vLLM 可能在启动阶段访问 Hugging Face 文件列表；网络异常时会出现 `Error retrieving file list: [Errno 99] Cannot assign requested address`。本地 snapshot 路径可以绕过这一步。

再在另一张卡启动 GRPO：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
  python -m wh_train grpo --config config/grpo.yaml
```

`PYTORCH_ALLOC_CONF=expandable_segments:True` 用于缓解 PyTorch CUDA 显存碎片导致的 OOM，不会降低训练本身需要的显存。若仍然 OOM，优先调小 `config/grpo.yaml` 中的 `num_generations`、`per_device_train_batch_size` 或 `max_completion_length`。

或使用 Makefile：

```bash
make grpo
```

GRPO 完成后会生成 `output/qwen35_grpo/`。评估 GRPO adapter 需要切回 SFT/eval 使用的 `train` 环境，因为 `make eval` 调用的是 LLaMA-Factory：

```bash
conda activate train
cd /home/catlab/wh_train
make eval ADAPTER=output/qwen35_grpo
make eval GPU=0 ADAPTER=output/qwen35_grpo

```

将 GRPO 评估结果与 SFT adapter 的 `output/qwen35_lora_vllm/predict/eval_report.json` 对比后，再决定是否替换部署 adapter。

---

## 目录职责

### `data/`

`data/dataset_info.json` 是 LLaMA-Factory 的数据集注册文件，训练配置依赖它，建议保留。

`data/train.jsonl` 和 `data/val.jsonl` 是生成数据，不入库。删除后需要重新执行 `python -m wh_train gen-data` 或 `make data`。

### `output/`

`output/` 是训练产物目录，不入库。默认 adapter 路径是 `output/qwen35_lora_vllm/`，部署、评估、交互试用和 GRPO 起点都会引用它。

如果只清理代码仓库空间，可以删除 `output/`；如果还要使用当前训练好的 LoRA adapter，就不要删除。

### `scripts/`

`scripts/` 是辅助运维脚本，仍由 Makefile 调用：

```bash
make check       # scripts/healthcheck.py
make check-data  # scripts/check_data.py
make post-train  # scripts/post_train.py
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

或：

```bash
make test
```

---

## 数据场景说明

见 `data.md`，包含 7 类场景（显式标准指令 / 隐式行为推断 / 隐式紧急推断 / 字段缺失 / 多备件 / ASR 噪声 / 数量模糊）及输入输出示例。
