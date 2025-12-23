# Qwen3-VL-8B-Instruct AWQ 量化实施文档

本文档详细记录了使用 `llm-compressor` 框架对多模态大模型 `Qwen3-VL-8B-Instruct` 进行 AWQ（Activation-aware Weight Quantization）量化的全过程。

## 1. 环境准备

### 1.1 基础环境
- **操作系统**: Linux
- **GPU**: NVIDIA RTX 4080 SUPER (16GB VRAM)
- **Python**: 3.10+
- **CUDA**: 12.1+

### 1.2 核心依赖
安装量化所需的库及多模态处理工具：
```bash
pip install llmcompressor==0.9.0
pip install transformers==4.57.3 accelerate==1.12.0
pip install qwen-vl-utils torchvision
```

## 2. 资源获取

### 2.1 模型下载
使用 `modelscope` 从魔塔社区下载原始 `Qwen3-VL-8B-Instruct` 模型：
- **脚本**: `src/scripts/download_model.py`
- **保存路径**: `./Qwen3-VL-8B-Instruct`

### 2.2 校准数据集
AWQ 量化需要少量数据进行激活值校准。
- **数据集**: `neuralmagic/calibration` (子集: LLM)
- **下载脚本**: `src/scripts/download_dataset.py`
- **本地路径**: `./calibration_dataset`

## 3. 量化实施

### 3.1 量化策略
- **算法**: AWQ (4-bit Group-wise)
- **配置**:
  - `num_bits`: 4
  - `group_size`: 128
  - `observer`: MSE (均方误差最小化)
- **保护策略**: 
  - 仅量化语言模型（Language Model）部分的 Linear 层。
  - **忽略视觉编码器 (Visual Encoder)**：视觉部分通常对量化敏感，通过正则 `re:model[.]visual.*` 保持原始精度，以保护模型的多模态理解能力。
  - **忽略特殊层**: `embed_tokens` 和 `lm_head` 保持高精度。

### 3.2 关键代码实现 (`src/scripts/quantize_qwen_vl.py`)
由于 Qwen3-VL 是较新的架构，我们手动定义了 `AWQMapping` 以确保层级匹配正确：
```python
mappings = [
    AWQMapping(smooth_layer="re:.*input_layernorm", balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"]),
    AWQMapping(smooth_layer="re:.*post_attention_layernorm", balance_layers=["re:.*gate_proj", "re:.*up_proj"]),
    AWQMapping(smooth_layer="re:.*up_proj", balance_layers=["re:.*down_proj"])
]
```

### 3.3 执行量化
```bash
python src/scripts/quantize_qwen_vl.py
```

## 4. 结果验证

### 4.1 存储对比
| 指标 | 原始模型 (BF16) | 量化模型 (AWQ INT4) |
| :--- | :--- | :--- |
| 总大小 | ~16.5 GB | **6.8 GB** |
| 文件结构 | 4 个 .safetensors 分片 | 2 个 .safetensors 分片 |

### 4.2 结构验证
通过 `src/scripts/inspect_model.py` 验证：
- **语言层**: 成功转换为 `CompressedLinear` 类型。
- **视觉层**: 保持为原始 `Linear` 类型，符合预期。

## 5. 结论
本次量化工作成功将 `Qwen3-VL-8B-Instruct` 压缩至约 6.8GB，在大幅降低显存占用的同时，通过有针对性的层级保护策略（保护视觉塔）最大程度保留了模型的多模态性能。
