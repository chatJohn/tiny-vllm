import torch
import torch.nn as nn
from typing import Tuple, Optional


def quantize_weight_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将权重量化为int8格式
    
    Args:
        weight: 原始权重张量
        
    Returns:
        (quantized_weight, scale): 量化后的int8权重和缩放因子
    """
    # 计算每个输出通道的最大绝对值
    max_vals = weight.abs().max(dim=-1, keepdim=True)[0]
    # 计算缩放因子
    scale = max_vals / 127.0
    scale[scale == 0] = 1.0  # 避免除零
    
    # 量化到int8
    quantized_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    
    return quantized_weight, scale


def quantize_weight_int4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """将权重量化为int4格式
    
    Args:
        weight: 原始权重张量
        
    Returns:
        (packed_weight, scale): 打包的int4权重和缩放因子
    """
    # 计算每个输出通道的最大绝对值
    max_vals = weight.abs().max(dim=-1, keepdim=True)[0]
    # 计算缩放因子
    scale = max_vals / 7.0  # int4范围是[-8, 7]
    scale[scale == 0] = 1.0  # 避免除零
    
    # 量化到int4并打包
    quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
    
    # 将int4打包为int8（每2个int4打包成1个int8）
    quantized = quantized + 8  # 映射到[0, 15]
    packed_weight = torch.zeros(quantized.shape[0], (quantized.shape[1] + 1) // 2, dtype=torch.uint8)
    
    for i in range(0, quantized.shape[1], 2):
        if i + 1 < quantized.shape[1]:
            packed_weight[:, i//2] = (quantized[:, i] << 4) | quantized[:, i+1]
        else:
            packed_weight[:, i//2] = quantized[:, i] << 4
    
    return packed_weight, scale


def dequantize_weight_int8(quantized_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """将int8权重反量化为原始格式
    
    Args:
        quantized_weight: int8量化权重
        scale: 缩放因子
        
    Returns:
        反量化后的权重
    """
    return quantized_weight.to(scale.dtype) * scale


def dequantize_weight_int4(packed_weight: torch.Tensor, scale: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """将int4权重反量化为原始格式
    
    Args:
        packed_weight: 打包的int4权重
        scale: 缩放因子
        original_shape: 原始权重形状
        
    Returns:
        反量化后的权重
    """
    # 解包int4
    quantized = torch.zeros(original_shape[0], original_shape[1], dtype=torch.int8)
    
    for i in range(0, original_shape[1], 2):
        if i + 1 < original_shape[1]:
            quantized[:, i] = (packed_weight[:, i//2] >> 4) & 0x0F
            quantized[:, i+1] = packed_weight[:, i//2] & 0x0F
        else:
            quantized[:, i] = (packed_weight[:, i//2] >> 4) & 0x0F
    
    # 映射回[-8, 7]
    quantized = quantized - 8
    
    return quantized.to(scale.dtype) * scale


class QuantizedLinear(nn.Module):
    """量化线性层基类"""
    
    def __init__(self, quantization: str):
        super().__init__()
        self.quantization = quantization
        self.quantized_weight = None
        self.scale = None
        self.original_shape = None
    
    def quantize_and_store(self, weight: torch.Tensor):
        """量化并存储权重"""
        self.original_shape = weight.shape
        
        if self.quantization == "int8":
            self.quantized_weight, self.scale = quantize_weight_int8(weight)
        elif self.quantization == "int4":
            self.quantized_weight, self.scale = quantize_weight_int4(weight)
        else:
            raise ValueError(f"不支持的量化类型: {self.quantization}")
    
    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        if self.quantization == "int8":
            return dequantize_weight_int8(self.quantized_weight, self.scale)
        elif self.quantization == "int4":
            return dequantize_weight_int4(self.quantized_weight, self.scale, self.original_shape)
        else:
            return self.quantized_weight  # float16直接返回


def create_quantized_linear(base_linear: nn.Module, quantization: str) -> nn.Module:
    """创建量化线性层包装器"""
    
    class QuantizedLinearWrapper(QuantizedLinear):
        def __init__(self, base_linear: nn.Module, quantization: str):
            super().__init__(quantization)
            self.base_linear = base_linear
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.quantization in ["int8", "int4"]:
                # 在推理时反量化权重
                dequantized_weight = self.dequantize_weight()
                # 使用反量化后的权重进行计算
                return torch.nn.functional.linear(x, dequantized_weight, self.base_linear.bias)
            else:
                # float16直接使用原始权重
                return self.base_linear(x)
    
    return QuantizedLinearWrapper(base_linear, quantization)