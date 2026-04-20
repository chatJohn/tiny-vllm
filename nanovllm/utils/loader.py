import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from .quantization import create_quantized_linear


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, quantization: str = "float16"):
    """加载模型权重，支持量化
    
    Args:
        model: 要加载权重的模型
        path: 模型路径
        quantization: 量化类型，可选"float16", "int8", "int4"
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 如果使用量化，包装所有线性层
    if quantization in ["int8", "int4"]:
        quantize_model_linear_layers(model, quantization)
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                print(f"{weight_name} {f.get_tensor(weight_name).shape}")
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
    
    # 如果使用量化，对加载的权重进行量化
    if quantization in ["int8", "int4"]:
        quantize_model_weights(model, quantization)


def quantize_model_linear_layers(model: nn.Module, quantization: str):
    """将模型中的线性层替换为量化版本"""
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, LinearBase)):
            # 替换为量化版本
            quantized_module = create_quantized_linear(module, quantization)
            setattr(model, name, quantized_module)
        else:
            # 递归处理子模块
            quantize_model_linear_layers(module, quantization)


def quantize_model_weights(model: nn.Module, quantization: str):
    """对模型权重进行量化"""
    for module in model.modules():
        if hasattr(module, 'quantize_and_store') and hasattr(module, 'weight'):
            module.quantize_and_store(module.weight.data)
            # 释放原始权重内存
            del module.weight
            module.weight = None


def print_model(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                print(f"{weight_name} {f.get_tensor(weight_name).shape}")


if __name__ == "__main__":
    import argparse

    argparse = argparse.ArgumentParser(description="nano vllm")
    argparse.add_argument(
        "--model-path", type=str, default="/nfs/ofs-llab-cold/model/Qwen/Qwen3-0.6B"
    )
    args = argparse.parse_args()
    print_model(args.model_path)
