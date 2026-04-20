"""GPTQ (Gradient-based Post-Training Quantization) implementation.

Reference:
    Frantar et al., "GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers", 2022. https://arxiv.org/abs/2210.17323

Core idea
---------
For a linear layer ``Y = X W^T`` we want to find an integer weight ``W_q`` that
minimizes ``||W X^T - W_q X^T||_F^2``.  Given a Hessian ``H = 2 X X^T`` (from
calibration activations), GPTQ quantizes the weights one column at a time in a
specific order and uses ``H^{-1}`` to spread the introduced error across the
remaining (not-yet-quantized) columns. This yields significantly smaller
quantization error than naive round-to-nearest (RTN).

This module provides:

* :func:`gptq_quantize_weight`    -- the core algorithm for a single weight.
* :class:`GPTQLinear`             -- a drop-in replacement for a linear layer
                                     that stores packed INT4 weights plus
                                     per-group scales / zero points and runs
                                     inference via a dequantize-then-matmul
                                     path (clear and dependency-free).
* :func:`apply_gptq_to_module`    -- recursively replace every linear-like
                                     module in a model with :class:`GPTQLinear`
                                     using GPTQ on its already-loaded weight.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core GPTQ algorithm
# ---------------------------------------------------------------------------
def _quantize_group(
    w: torch.Tensor, bits: int, sym: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scale / zero_point for a 2-D weight block ``w`` of shape
    ``(out_features, group_size)`` and return ``(q, scale, zero)``.

    * asymmetric: q in [0, 2^bits - 1]
    * symmetric : q in [-(2^(bits-1)), 2^(bits-1) - 1]
    """
    qmax = (1 << bits) - 1  # asymmetric max
    if sym:
        absmax = w.abs().amax(dim=1, keepdim=True)
        absmax = torch.clamp(absmax, min=1e-8)
        scale = absmax / ((1 << (bits - 1)) - 1)
        zero = torch.full_like(scale, 1 << (bits - 1))
        q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    else:
        wmax = w.amax(dim=1, keepdim=True)
        wmin = w.amin(dim=1, keepdim=True)
        scale = (wmax - wmin) / qmax
        scale = torch.clamp(scale, min=1e-8)
        zero = torch.round(-wmin / scale)
        q = torch.clamp(torch.round(w / scale) + zero, 0, qmax)
    return q, scale, zero


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: Optional[torch.Tensor] = None,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the GPTQ algorithm to a single weight matrix.

    Parameters
    ----------
    weight : (out_features, in_features) float tensor
        The weight of the linear layer (the row dim is "out" like ``nn.Linear``).
    hessian : (in_features, in_features) float tensor, optional
        The Hessian matrix ``H = 2 X X^T`` accumulated over calibration inputs.
        When ``None`` we fall back to ``H = I`` which reduces GPTQ to
        round-to-nearest with error compensation disabled (still useful for a
        working pipeline and can be replaced by a real Hessian later).
    bits : int
        Number of bits, 4 by default (common GPTQ setting).
    group_size : int
        Size of each quantization group along the in-features dim.
    sym : bool
        Whether to use symmetric quantization.
    percdamp : float
        Percent of the mean diagonal of ``H`` to add as damping before
        Cholesky decomposition, for numerical stability.
    blocksize : int
        Column-block size used during the iteration.

    Returns
    -------
    q_weight : (out_features, in_features) int tensor
        Quantized integer weights (stored in ``torch.int32`` for simplicity;
        actual values fit in ``bits`` bits).
    scales   : (out_features, num_groups) float tensor
    zeros    : (out_features, num_groups) float tensor
    """
    assert weight.dim() == 2, "GPTQ expects a 2-D weight"
    out_features, in_features = weight.shape
    device = weight.device
    # use fp32 for numerical stability of the linear algebra
    W = weight.detach().to(torch.float32).clone()

    if hessian is None:
        H = torch.eye(in_features, device=device, dtype=torch.float32)
    else:
        H = hessian.to(device=device, dtype=torch.float32).clone()

    # --- Handle "dead" input columns (never activated) ----------------------
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # --- Add damping on the diagonal ----------------------------------------
    damp = percdamp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(in_features, device=device)
    H[diag_idx, diag_idx] += damp

    # --- Cholesky factorization of H^{-1} -----------------------------------
    # H is symmetric positive-definite; we compute H^{-1} and then take the
    # upper-triangular Cholesky of the inverse. This exactly matches the
    # original GPTQ implementation.
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        H_inv = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        # Fall back to identity if decomposition fails.
        H_inv = torch.eye(in_features, device=device, dtype=torch.float32)

    assert in_features % group_size == 0, (
        f"in_features {in_features} must be divisible by group_size {group_size}"
    )
    num_groups = in_features // group_size

    Q = torch.zeros_like(W)
    Losses = torch.zeros_like(W)
    scales = torch.zeros(out_features, num_groups, device=device, dtype=torch.float32)
    zeros = torch.zeros(out_features, num_groups, device=device, dtype=torch.float32)

    # --- Iterate over column blocks ----------------------------------------
    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = H_inv[i1:i2, i1:i2]

        for j in range(count):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]

            # Start of a new quantization group -> recompute scale/zero
            if col % group_size == 0:
                grp_end = min(col + group_size, in_features)
                block = W[:, col:grp_end]
                q_block, s_block, z_block = _quantize_group(block, bits, sym)
                g_idx = col // group_size
                scales[:, g_idx] = s_block.squeeze(1)
                zeros[:, g_idx] = z_block.squeeze(1)

            g_idx = col // group_size
            s = scales[:, g_idx]
            z = zeros[:, g_idx]
            qmax = (1 << bits) - 1

            q = torch.clamp(torch.round(w / s) + z, 0, qmax)
            w_q = (q - z) * s

            Q1[:, j] = q
            Losses[:, i1 + j] = (w - w_q) ** 2 / d ** 2

            err1 = (w - w_q) / d
            W1[:, j:] -= err1.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
            Err1[:, j] = err1

        Q[:, i1:i2] = Q1
        # Propagate the aggregated error of this block to all remaining columns
        W[:, i2:] -= Err1 @ H_inv[i1:i2, i2:]

    q_weight = Q.to(torch.int32)
    return q_weight, scales, zeros


# ---------------------------------------------------------------------------
# INT4 packing helpers (2 weights per byte along the in-features dim)
# ---------------------------------------------------------------------------
def pack_int4(q_weight: torch.Tensor) -> torch.Tensor:
    """Pack a ``(out_features, in_features)`` int32 tensor with values in
    ``[0, 15]`` into ``(out_features, in_features // 2)`` uint8."""
    assert q_weight.dim() == 2
    out_features, in_features = q_weight.shape
    assert in_features % 2 == 0, "in_features must be even for int4 packing"
    low = (q_weight[:, 0::2] & 0x0F).to(torch.uint8)
    high = (q_weight[:, 1::2] & 0x0F).to(torch.uint8)
    return (high << 4) | low


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`.  Returns int32 with values in ``[0, 15]``."""
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    out = torch.empty(packed.shape[0], in_features, dtype=torch.int32, device=packed.device)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


# ---------------------------------------------------------------------------
# Quantized linear layer (dequantize-then-matmul, reference implementation)
# ---------------------------------------------------------------------------
class GPTQLinear(nn.Module):
    """A drop-in replacement for linear-like layers after GPTQ quantization.

    We keep everything simple and framework-agnostic: the packed INT4 weights
    are unpacked + dequantized to the compute dtype every forward pass. This
    is significantly slower than a fused INT4 kernel but matches full-precision
    numerics closely and keeps the implementation portable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        group_size: int,
        bits: int = 4,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        assert bits == 4, "Current GPTQLinear only supports 4-bit"
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        num_groups = in_features // group_size

        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, num_groups, dtype=compute_dtype),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(out_features, num_groups, dtype=compute_dtype),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype))
        else:
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # Loading & forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def load_from_float_weight(
        self,
        fp_weight: torch.Tensor,
        fp_bias: Optional[torch.Tensor] = None,
        hessian: Optional[torch.Tensor] = None,
    ) -> None:
        """Quantize an already-loaded float weight using GPTQ and store it."""
        q, s, z = gptq_quantize_weight(
            fp_weight,
            hessian=hessian,
            bits=self.bits,
            group_size=self.group_size,
            sym=False,
        )
        packed = pack_int4(q).to(self.qweight.device)
        self.qweight.copy_(packed)
        self.scales.copy_(s.to(self.scales.dtype).to(self.scales.device))
        self.zeros.copy_(z.to(self.zeros.dtype).to(self.zeros.device))
        if self.bias is not None and fp_bias is not None:
            self.bias.data.copy_(fp_bias.to(self.bias.dtype))
        # Release large temporaries before we return, so the caller's
        # empty_cache() call has something to actually hand back.
        del q, s, z, packed

    def dequantize(self) -> torch.Tensor:
        """Return the dequantized weight as ``(out_features, in_features)``."""
        q = unpack_int4(self.qweight, self.in_features)  # int32 in [0, 15]
        # Expand scale/zero per column by repeating each group.
        s = self.scales.repeat_interleave(self.group_size, dim=1)
        z = self.zeros.repeat_interleave(self.group_size, dim=1)
        w = (q.to(self.compute_dtype) - z) * s
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        return F.linear(x, w, self.bias)


# ---------------------------------------------------------------------------
# Helpers to swap linear-like modules in-place
# ---------------------------------------------------------------------------
def _get_linear_info(module: nn.Module) -> Optional[tuple[int, int, bool]]:
    """Return ``(in_features, out_features, has_bias)`` if ``module`` is a
    linear-like layer we should quantize, otherwise ``None``.

    We detect both ``nn.Linear`` and the ``LinearBase`` subclasses defined in
    :mod:`nanovllm.layers.linear` (which all expose a 2-D ``weight`` of shape
    ``(out, in)``).
    """
    weight = getattr(module, "weight", None)
    if weight is None or not torch.is_tensor(weight.data):
        return None
    if weight.data.dim() != 2:
        return None
    # Skip obvious non-linear modules that also have 2-D weights (e.g.
    # embeddings).  They don't expose ``in_features`` / ``out_features``.
    if not (hasattr(module, "input_size") or hasattr(module, "in_features")):
        return None
    if hasattr(module, "input_size"):
        # nanovllm LinearBase subclasses store weight as (out_per_partition, in)
        out_features, in_features = weight.data.shape
    else:
        in_features = module.in_features
        out_features = module.out_features
    has_bias = getattr(module, "bias", None) is not None
    return in_features, out_features, has_bias


def apply_gptq_to_module(
    root: nn.Module,
    group_size: int = 128,
    bits: int = 4,
    compute_dtype: Optional[torch.dtype] = None,
    verbose: bool = False,
) -> int:
    """Recursively replace every linear-like child of ``root`` with a
    :class:`GPTQLinear` whose weight is the GPTQ-quantized version of the
    original (already-loaded) float weight.

    Returns the number of layers that were replaced.

    Notes
    -----
    After each replacement we *immediately* drop the references to the old
    float weight/bias and call ``torch.cuda.empty_cache()``. Without this
    step PyTorch's caching allocator keeps the original fp16/bf16 weight
    block alive (plus any transient fp32 buffers used during quantization),
    and the process-level GPU memory reported by ``nvidia-smi`` never
    actually decreases even though INT4 packing is correct.
    """
    replaced = 0

    def _recurse(parent: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            info = _get_linear_info(child)
            # Only quantize layers whose in_features is divisible by group_size.
            if info is not None:
                in_features, out_features, has_bias = info
                if in_features % group_size != 0:
                    if verbose:
                        print(
                            f"[GPTQ] skip {full}: in_features {in_features} "
                            f"not divisible by group_size {group_size}"
                        )
                elif type(child).__name__ in {
                    "VocabParallelEmbedding",
                    "ParallelLMHead",
                }:
                    # Skip embeddings / LM head for safety: they are very
                    # sensitive to quantization and often kept in fp.
                    if verbose:
                        print(f"[GPTQ] skip {full}: embedding/lm_head")
                else:
                    dtype = compute_dtype or child.weight.data.dtype
                    device = child.weight.data.device
                    new_layer = GPTQLinear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=has_bias,
                        group_size=group_size,
                        bits=bits,
                        compute_dtype=dtype,
                    )
                    new_layer.to(device)
                    bias_data = child.bias.data if has_bias else None
                    new_layer.load_from_float_weight(child.weight.data, bias_data)
                    setattr(parent, name, new_layer)

                    # --- Critical: release the original float weight NOW.
                    # We explicitly null out the tensor on the old module so
                    # there is no lingering reference (the child object
                    # itself will be GC'd once this local frame unwinds).
                    try:
                        child.weight = None  # type: ignore[assignment]
                    except Exception:
                        pass
                    if has_bias:
                        try:
                            child.bias = None  # type: ignore[assignment]
                        except Exception:
                            pass
                    del child, bias_data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    replaced += 1
                    if verbose:
                        print(
                            f"[GPTQ] quantized {full}  "
                            f"(in={in_features}, out={out_features})"
                        )
                    # do NOT recurse into the replaced layer
                    continue
            _recurse(child, full)

    _recurse(root)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced
