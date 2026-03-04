# Multiple models adapted for nano-vllm
## 🪄 Features:
- Multiple Models Adapt: [Llama2](https://huggingface.co/meta-llama/Llama-2-7b), [Qwen2](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3MOE](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- Cuda Kernels Rebuild: activation.cu, embedding.cu, layernorm.cu, linear.cu, rotary_embedding.cu
- Tensor Parallel Fit: Achieve the ***TP***, so you can use this project to infer your model in multiple GPUs
- Inherit vLLM Features: PageAttention, Continuous Batching, Chunked Prefill, Prefix Caching...
## 🪜 Structure
tiny-vllm 作为 vllm 轻量化学习型项目，代码结构精简且聚焦 LLM 推理核心链路，整体目录组织如下：

```plaintext
tiny-vllm/
├── nanovllm/                # 核心轻量化LLM推理模块（基于vllm核心逻辑拆解+轻量化改造）
│   ├── engine               # 推理框架核心引擎，包括pageattn等
│   ├── layers               # 模型的核心部分，包括pytorch实现层和cuda实现的部分算子
│   ├── models               # 不同的模型的适配
│   ├── __init__.py          # 初始化文件
│   ├── config.py            # 配置文件
│   ├── llm.py               # LLM接口
│   ├── sampling_params.py   # 采样参数
│   ├── model_executor.py    # 模型推理执行核心（封装前向计算、张量处理、设备适配逻辑）
│   └── utils                # 工具
├── example.py               # 功能验证示例（单卡推理/对话交互演示，验证核心模块可用性）
├── requirements.txt         # 项目依赖清单（torch/transformers等，适配轻量化部署）
└── README.md                # 项目说明文档（背景、代码结构、运行步骤、核心实现说明）

```
## ⚒️ Requirements
 - Python 3.8+
 - CUDA 11.0+
 - Pytorch 2.1.0
 - Transformer
 - Flash Attention
