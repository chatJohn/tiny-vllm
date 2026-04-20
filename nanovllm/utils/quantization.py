"""Quantization dispatch utilities.

This module is the public entry-point for all quantization methods in
``nanovllm``.  Each ``quant_method`` (e.g. ``"gptq"``) is a self-contained
implementation living in its own module; here we only expose a thin API that
the model loader can call without caring about the concrete algorithm.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .gptq import apply_gptq_to_module


SUPPORTED_QUANT_METHODS = (None, "gptq")


def apply_quantization(
    model: nn.Module,
    quant_method: Optional[str],
    *,
    group_size: int = 128,
    bits: int = 4,
    compute_dtype: Optional[torch.dtype] = None,
    verbose: bool = False,
) -> int:
    """Post-load quantization dispatcher.

    ``quant_method=None`` means the model stays in its loaded (floating-point)
    precision and nothing is changed.  Returns the number of layers that were
    replaced with a quantized version.
    """
    if quant_method is None:
        return 0
    elif quant_method == "gptq":
        return apply_gptq_to_module(
            model,
            group_size=group_size,
            bits=bits,
            compute_dtype=compute_dtype,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unsupported quant_method: {quant_method!r}. "
            f"Supported methods: {SUPPORTED_QUANT_METHODS}"
        )