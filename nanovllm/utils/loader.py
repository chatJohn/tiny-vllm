import os
from glob import glob
from typing import Optional

import torch
from safetensors import safe_open
from torch import nn

from .quantization import apply_quantization


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module,
    path: str,
    quant_method: Optional[str] = None,
    *,
    quant_group_size: int = 128,
    quant_bits: int = 4,
):
    """Load weights into ``model`` and optionally apply a post-training
    quantization scheme.

    Parameters
    ----------
    model : nn.Module
        The model skeleton (with float weights allocated).
    path : str
        Directory containing ``*.safetensors`` weight files.
    quant_method : Optional[str]
        ``None`` for full-precision, ``"gptq"`` to apply GPTQ int4 quantization
        after the float weights are loaded.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
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

    # Apply quantization (if requested) once all float weights are loaded.
    if quant_method is not None:
        num_replaced = apply_quantization(
            model,
            quant_method=quant_method,
            group_size=quant_group_size,
            bits=quant_bits,
            verbose=True,
        )
        print(
            f"[load_model] applied {quant_method} quantization to "
            f"{num_replaced} linear layers"
        )
        # Drop leftover GPU memory from the now-replaced float weights.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
