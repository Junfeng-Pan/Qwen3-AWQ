# Qwen3-VL-8B-Instruct AWQ 量化实战

本项目实现了对多模态大模型 **Qwen3-VL-8B-Instruct** 的 AWQ (4-bit) 量化。通过使用 `llm-compressor` 框架，我们在大幅降低显存占用（~16GB -> ~7GB）的同时，通过特定的层级保护策略，最大程度保留了模型的视觉理解能力。

## 📂 目录结构

```text
.
├── datasets/                 # 存放下载的数据集 (calibration, OCRBench)
├── docs/                     # 项目文档 (量化实施文档, 评测报告)
├── Qwen3-VL-8B-Instruct/     # 原始模型 (需下载)
├── Qwen3-VL-8B-Instruct-AWQ/ # 量化后模型输出目录
├── src/
│   ├── download/             # 资源下载脚本
│   │   ├── download_model.py    # 下载 Qwen3 模型
│   │   ├── download_dataset.py  # 下载校准数据
│   │   └── download_ocrbench.py # 下载评测数据
│   ├── scripts/              # 核心功能脚本
│   │   ├── quantize_qwen_vl.py  # AWQ 量化主脚本
│   │   ├── benchmark.py         # 速度/显存基准测试
│   │   └── inspect_model.py     # 模型结构检查
│   ├── tests/                # 测试与评测脚本
│   │   ├── test_quantized_model.py # 单图推理测试
│   │   ├── eval_ocrbench_custom.py # OCRBench 评测
│   │   └── analyze_ocrbench.py     # 评测结果分析
│   └── outputs/              # 评测结果输出
└── requirements.txt          # 环境依赖
```

## 🛠️ 环境搭建

本项目基于 Python 3.10+ 和 PyTorch 环境。

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *建议显式安装 `flash-attn` 以获得更快的训练/推理速度（可选）。*

## 🚀 快速开始

### 1. 准备资源
运行以下命令下载原始模型和量化所需的校准数据集：

```bash
# 下载模型 (使用 ModelScope)
python src/download/download_model.py

# 下载校准数据集 (neuralmagic/calibration)
python src/download/download_dataset.py

# (可选) 下载 OCRBench 评测集
python src/download/download_ocrbench.py
```

### 2. 执行量化
运行量化脚本。该脚本会自动加载模型，应用 AWQ 算法（W4A16），并保存到 `Qwen3-VL-8B-Instruct-AWQ` 目录。

**策略重点**：脚本已配置为**不量化 Vision Encoder**，以保护多模态性能。同时针对 Qwen3-VL 架构手动修复了层级映射（Layer Mapping）。

```bash
python src/scripts/quantize_qwen_vl.py
```

### 3. 测试与验证

*   **简单的单图推理测试**:
    ```bash
    python src/tests/test_quantized_model.py
    ```

*   **基准测试 (速度/显存)**:
    ```bash
    python src/scripts/benchmark.py --model_path Qwen3-VL-8B-Instruct-AWQ
    ```

### 4. 效果评测 (OCRBench)
对量化模型进行 OCR 能力评估：

```bash
# 1. 运行评测 (可能需要一段时间)
python src/tests/eval_ocrbench_custom.py

# 2. 查看分析报告
python src/tests/analyze_ocrbench.py
```

## 📊 结果摘要

| 指标 | 原始模型 (BF16) | 量化模型 (AWQ INT4) | 备注 |
| :--- | :--- | :--- | :--- |
| **模型大小** | ~16.5 GB | **6.8 GB** | 📉 减少约 59% |
| **显存占用** | > 16 GB | **~7.5 GB** | 适合单卡 8G/12G 部署 |
| **OCRBench** | - | **87.4%** | 保持了极高的视觉识别精度 |

详细评测结果请参阅 [docs/量化模型效果评测.md](docs/量化模型效果评测.md)。

## 📝 技术细节

*   **量化框架**: `llm-compressor` (vLLM Project)
*   **量化配置**:
    *   Group Size: 128
    *   Bits: 4
    *   Observer: MSE
*   **架构适配**: 针对 `Qwen3-VL` 架构，手动定义了 `AWQMapping` 以解决自动推断层级关系失败的问题，并移除了不兼容的 `v_proj` 映射。
