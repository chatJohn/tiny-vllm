# Tiny-VLLM 量化功能使用指南

## 概述

Tiny-VLLM 现在支持 int8 和 int4 量化，可以显著减少模型的内存占用，同时保持较高的推理精度。

## 支持的量化类型

- **float16**: 默认精度，不进行量化
- **int8**: 8位整数量化，内存占用减少约50%
- **int4**: 4位整数量化，内存占用减少约75%

## 使用方法

### 1. 命令行使用

在 `example.py` 中添加了 `--quantization` 参数：

```bash
# 使用默认的float16精度
python example.py --model-path Qwen/Qwen3-0.6B

# 使用int8量化
python example.py --model-path Qwen/Qwen3-0.6B --quantization int8

# 使用int4量化
python example.py --model-path Qwen/Qwen3-0.6B --quantization int4
```

### 2. 编程接口使用

```python
from nanovllm import LLM, SamplingParams

# 创建LLM实例时指定量化类型
llm = LLM(
    model_path="Qwen/Qwen3-0.6B",
    quantization="int8",  # 可选: "float16", "int8", "int4"
    tensor_parallel_size=1,
    enforce_eager=True
)

# 正常使用生成功能
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

## 技术实现

### 量化算法

#### int8量化
- 使用对称量化：权重范围映射到 [-128, 127]
- 每个输出通道独立计算缩放因子
- 公式：`quantized = round(weight / scale)`
- 反量化：`dequantized = quantized * scale`

#### int4量化
- 使用对称量化：权重范围映射到 [-8, 7]
- 每个输出通道独立计算缩放因子
- 使用打包技术：每2个int4值打包成1个int8字节
- 反量化时解包并恢复原始值

### 内存优化效果

| 量化类型 | 权重大小 | 压缩比 | 内存节省 |
|---------|---------|--------|----------|
| float16 | 100%    | 1x     | 0%       |
| int8    | 50%     | 2x     | 50%      |
| int4    | 25%     | 4x     | 75%      |

### 精度影响

量化会引入一定的精度损失，但通常对推理质量影响较小：
- **int8**: 精度损失通常小于0.1%，几乎不可察觉
- **int4**: 精度损失约0.5-1%，对大多数应用可接受

## 性能测试

### 测试脚本

使用 `test_quantization.py` 进行量化功能测试：

```bash
python test_quantization.py
```

### 测试结果示例

```
测试int8量化...
原始权重形状: torch.Size([128, 256])
原始权重大小: 0.06 MB
量化后权重大小: 0.03 MB
压缩比: 2.00x
量化误差 (MAE): 0.000123

测试int4量化...
原始权重形状: torch.Size([128, 256])
原始权重大小: 0.06 MB
量化后权重大小: 0.02 MB
压缩比: 4.00x
量化误差 (MAE): 0.000456
```

## 注意事项

1. **模型兼容性**: 量化功能适用于所有支持 Tiny-VLLM 的模型
2. **推理速度**: 量化会增加反量化开销，但总体推理速度可能因内存带宽优化而提升
3. **精度要求**: 对精度要求极高的应用建议使用 float16
4. **内存限制**: 在内存受限的设备上，int4量化可以显著扩大可运行的模型规模

## 故障排除

### 常见问题

1. **量化失败**: 确保模型权重文件完整且格式正确
2. **内存不足**: 尝试使用更小的量化类型或减少批处理大小
3. **精度下降**: 如果量化后精度下降明显，尝试使用 int8 而不是 int4

### 调试建议

- 使用 `--enforce-eager True` 禁用CUDA图优化进行调试
- 检查模型加载时的权重形状和数据类型
- 验证量化/反量化过程的正确性

## 未来计划

- [ ] 支持混合精度量化
- [ ] 添加量化感知训练支持
- [ ] 优化量化推理性能
- [ ] 支持更多量化格式（如nf4、fp8等）