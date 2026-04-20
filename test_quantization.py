#!/usr/bin/env python3
"""
量化功能测试脚本
测试int8和int4量化是否正常工作
"""

import torch
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanovllm.utils.quantization import (
    quantize_weight_int8, 
    quantize_weight_int4, 
    dequantize_weight_int8, 
    dequantize_weight_int4
)


def test_int8_quantization():
    """测试int8量化功能"""
    print("测试int8量化...")
    
    # 创建测试权重
    original_weight = torch.randn(128, 256, dtype=torch.float16)
    print(f"原始权重形状: {original_weight.shape}")
    print(f"原始权重大小: {original_weight.element_size() * original_weight.nelement() / 1024 / 1024:.2f} MB")
    
    # 量化
    quantized_weight, scale = quantize_weight_int8(original_weight)
    print(f"量化后权重形状: {quantized_weight.shape}")
    print(f"量化后权重大小: {quantized_weight.element_size() * quantized_weight.nelement() / 1024 / 1024:.2f} MB")
    print(f"缩放因子形状: {scale.shape}")
    
    # 反量化
    dequantized_weight = dequantize_weight_int8(quantized_weight, scale)
    print(f"反量化后权重形状: {dequantized_weight.shape}")
    
    # 计算误差
    error = torch.abs(original_weight - dequantized_weight).mean()
    print(f"量化误差 (MAE): {error.item():.6f}")
    
    # 计算压缩比
    original_size = original_weight.element_size() * original_weight.nelement()
    quantized_size = quantized_weight.element_size() * quantized_weight.nelement() + scale.element_size() * scale.nelement()
    compression_ratio = original_size / quantized_size
    print(f"压缩比: {compression_ratio:.2f}x")
    
    print("int8量化测试完成!\n")


def test_int4_quantization():
    """测试int4量化功能"""
    print("测试int4量化...")
    
    # 创建测试权重
    original_weight = torch.randn(128, 256, dtype=torch.float16)
    print(f"原始权重形状: {original_weight.shape}")
    print(f"原始权重大小: {original_weight.element_size() * original_weight.nelement() / 1024 / 1024:.2f} MB")
    
    # 量化
    quantized_weight, scale = quantize_weight_int4(original_weight)
    print(f"量化后权重形状: {quantized_weight.shape}")
    print(f"量化后权重大小: {quantized_weight.element_size() * quantized_weight.nelement() / 1024 / 1024:.2f} MB")
    print(f"缩放因子形状: {scale.shape}")
    
    # 反量化
    dequantized_weight = dequantize_weight_int4(quantized_weight, scale, original_weight.shape)
    print(f"反量化后权重形状: {dequantized_weight.shape}")
    
    # 计算误差
    error = torch.abs(original_weight - dequantized_weight).mean()
    print(f"量化误差 (MAE): {error.item():.6f}")
    
    # 计算压缩比
    original_size = original_weight.element_size() * original_weight.nelement()
    quantized_size = quantized_weight.element_size() * quantized_weight.nelement() + scale.element_size() * scale.nelement()
    compression_ratio = original_size / quantized_size
    print(f"压缩比: {compression_ratio:.2f}x")
    
    print("int4量化测试完成!\n")


def test_quantization_performance():
    """测试量化性能"""
    print("测试量化性能...")
    
    # 测试不同大小的权重
    sizes = [(64, 64), (128, 256), (512, 512), (1024, 1024)]
    
    for size in sizes:
        print(f"\n测试权重大小: {size}")
        original_weight = torch.randn(size, dtype=torch.float16)
        
        # int8测试
        import time
        
        start_time = time.time()
        quantized_weight_int8, scale_int8 = quantize_weight_int8(original_weight)
        dequantized_weight_int8 = dequantize_weight_int8(quantized_weight_int8, scale_int8)
        int8_time = time.time() - start_time
        
        int8_error = torch.abs(original_weight - dequantized_weight_int8).mean()
        
        # int4测试
        start_time = time.time()
        quantized_weight_int4, scale_int4 = quantize_weight_int4(original_weight)
        dequantized_weight_int4 = dequantize_weight_int4(quantized_weight_int4, scale_int4, original_weight.shape)
        int4_time = time.time() - start_time
        
        int4_error = torch.abs(original_weight - dequantized_weight_int4).mean()
        
        print(f"int8 - 时间: {int8_time*1000:.2f}ms, 误差: {int8_error.item():.6f}")
        print(f"int4 - 时间: {int4_time*1000:.2f}ms, 误差: {int4_error.item():.6f}")


if __name__ == "__main__":
    print("开始量化功能测试...\n")
    
    test_int8_quantization()
    test_int4_quantization()
    test_quantization_performance()
    
    print("所有测试完成!")