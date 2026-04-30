import os
from glob import glob
from typing import Optional

import torch
from safetensors import safe_open
from torch import nn

from .quantization import apply_quantization


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _summarize_model_storage(model: nn.Module, tag: str) -> None:
    """Print a per-dtype storage breakdown of a model's parameters + buffers.

    This is extremely useful to verify the footprint change after a weight-only
    quantization pass: before GPTQ we expect ~all bytes to be fp16/bf16, after
    GPTQ the bulk should move into uint8 (packed INT4) with small fp16 scale /
    zero buffers per group.
    """
    totals: dict[str, list[int]] = {}
    linear_like = 0
    gptq_like = 0
    total_bytes = 0
    total_numel = 0

    for mod_name, module in model.named_modules():
        # Track structural layer counts for quick before/after comparison
        cls = type(module).__name__
        if cls == "GPTQLinear":
            gptq_like += 1
        elif hasattr(module, "weight") and torch.is_tensor(getattr(module, "weight", None)) \
                and module.weight is not None and module.weight.dim() == 2 \
                and (hasattr(module, "in_features") or hasattr(module, "input_size")):
            linear_like += 1

    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is None:
            continue
        dtype = str(t.dtype).replace("torch.", "")
        totals.setdefault(dtype, [0, 0])
        b = _tensor_bytes(t)
        totals[dtype][0] += t.numel()
        totals[dtype][1] += b
        total_numel += t.numel()
        total_bytes += b

    print(f"[load_model][{tag}] ---- weight storage summary ----")
    print(
        f"[load_model][{tag}] linear_layers(fp)={linear_like}  "
        f"gptq_linear_layers={gptq_like}  "
        f"total_numel={total_numel:,}  total_bytes={total_bytes/1024**2:.2f} MB"
    )
    for dtype, (nel, b) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        print(
            f"[load_model][{tag}]   dtype={dtype:<10} "
            f"numel={nel:>14,}  bytes={b/1024**2:>10.2f} MB"
        )


def load_model(
    model: nn.Module,
    path: str,
    quant_method: Optional[str] = None,
    *,
    quant_group_size: int = 128,
    quant_bits: int = 4,
    ep_rank: int = 0,
    ep_size: int = 1,
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
    ep_rank : int
        Expert parallel rank of this process (0 when EP is disabled).
    ep_size : int
        Total number of expert parallel ranks (1 when EP is disabled).
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # Determine which global expert indices belong to this EP rank.
    # Expert weights in safetensors follow the pattern:
    #   model.layers.{layer}.mlp.experts.{expert_idx}.{sub_weight}
    # When ep_size > 1 we skip experts that don't belong to this rank.
    import re
    _expert_re = re.compile(r"\.experts\.(\d+)\.")

    def _should_skip_expert(weight_name: str) -> bool:
        """Return True if this weight belongs to an expert not owned by this rank."""
        if ep_size <= 1:
            return False
        m = _expert_re.search(weight_name)
        if m is None:
            return False
        global_expert_idx = int(m.group(1))
        experts_per_rank = None
        # Try to infer experts_per_rank from the model
        for mod in model.modules():
            if hasattr(mod, "experts_per_rank"):
                experts_per_rank = mod.experts_per_rank
                break
        if experts_per_rank is None:
            return False
        owner_rank = global_expert_idx // experts_per_rank
        return owner_rank != ep_rank

    def _remap_expert_param_name(weight_name: str) -> str:
        """Remap global expert index to local expert index in param name."""
        if ep_size <= 1:
            return weight_name
        m = _expert_re.search(weight_name)
        if m is None:
            return weight_name
        global_expert_idx = int(m.group(1))
        experts_per_rank = None
        for mod in model.modules():
            if hasattr(mod, "experts_per_rank"):
                experts_per_rank = mod.experts_per_rank
                break
        if experts_per_rank is None:
            return weight_name
        local_expert_idx = global_expert_idx % experts_per_rank
        return weight_name[:m.start(1)] + str(local_expert_idx) + weight_name[m.end(1):]

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip expert weights that don't belong to this EP rank
                if _should_skip_expert(weight_name):
                    continue

                # Remap global expert index → local expert index in param name
                param_name = _remap_expert_param_name(weight_name)

                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        mapped_param_name = param_name.replace(k, v)
                        param = model.get_parameter(mapped_param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))


    # Apply quantization (if requested) once all float weights are loaded.
    # Always emit a "before" snapshot of the model's weight storage so that the
    # effect of quantization (or the lack of it) is clearly visible in the
    # terminal log.  Use torch.cuda.synchronize() + memory_allocated() to get
    # an accurate before/after GPU footprint.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_before = torch.cuda.memory_allocated()
    else:
        gpu_before = 0
    _summarize_model_storage(model, tag="before_quant")
    print(
        f"[load_model][before_quant] cuda.memory_allocated="
        f"{gpu_before/1024**2:.2f} MB"
    )

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
            f"{num_replaced} linear layers (group_size={quant_group_size}, "
            f"bits={quant_bits})"
        )
        # Drop leftover GPU memory from the now-replaced float weights.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_after = torch.cuda.memory_allocated()
        else:
            gpu_after = 0
        _summarize_model_storage(model, tag="after_quant")
        print(
            f"[load_model][after_quant]  cuda.memory_allocated="
            f"{gpu_after/1024**2:.2f} MB  "
            f"delta={(gpu_after - gpu_before)/1024**2:+.2f} MB"
        )
    else:
        print("[load_model] quant_method=None, model kept in full precision")


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
