# wh_train — 港口备件指令解析微调

基于 Qwen3.5-4B + LoRA，对港口备件领料指令进行微调，将自然语言指令解析为结构化 WorkOrder JSON 数组。

替换 `wh_graphrag_re` 中的 ReAct 多轮 Agent，推理延迟从 2–10 s 降至约 200 ms。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU  | NVIDIA RTX 3090 Ti 24 G（或同等显存） |
| CUDA | 12.2 |
| Python | 3.10 |

## 安装

```bash
# 1. 创建 conda 环境
conda create -n train python=3.10 -y
conda activate train

# 2. 安装 PyTorch（CUDA 12.4 wheel 兼容 CUDA 12.2 驱动）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# 3. 安装其余依赖
pip install -r requirements.txt
```

---

## 项目结构

```
wh_train/
├── config/
│   ├── train_config.yaml    # 训练超参数（ms-swift --config 加载）
│   └── export_config.yaml   # LoRA 导出合并配置
├── data/
│   ├── train.jsonl          # 训练集（ShareGPT 格式）
│   └── val.jsonl            # 验证集
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
swift sft --config config/train_config.yaml
```

所有超参数在 `config/train_config.yaml` 中配置，无需修改命令行。

训练日志与 checkpoint 保存至 `output/qwen35_lora/`。

### 3. 评估

```bash
python evaluate.py --val_file data/val.jsonl --model_dir output/qwen35_lora/checkpoint-best
```

输出四项指标：

| 指标 | 目标 |
|------|------|
| JSON 解析成功率 | > 98% |
| 字段准确率（各字段平均） | > 90% |
| 数组长度准确率（多备件） | > 85% |
| null 召回率（缺失字段） | > 95% |

### 4. 导出合并模型

```bash
swift export --config config/export_config.yaml
```

合并后的模型保存至 `output/qwen35_merged/`。

---

## 修改训练参数

编辑 `config/train_config.yaml`，关键参数说明：

```yaml
lora_rank: 16        # LoRA 秩，增大可提升表达能力但增加显存
learning_rate: 2e-4  # 初始学习率
num_train_epochs: 3  # 训练轮数
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # 等效 batch_size = 4 × 4 = 16
```

---

## 集成到 wh_graphrag_re

训练完成后，修改 `/home/catlab/wh/wh_graphrag_re/.env`：

```ini
VLM35_MODEL=./output/qwen35_merged
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
