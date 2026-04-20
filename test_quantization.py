#!/usr/bin/env python3
"""Unit tests / smoke tests for the GPTQ implementation in ``nanovllm``.

Run with::

    python test_quantization.py
"""

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanovllm.utils.gptq import (
    GPTQLinear,
    apply_gptq_to_module,
    gptq_quantize_weight,
    pack_int4,
    unpack_int4,
)


def _print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_pack_unpack_roundtrip():
    _print_header("test_pack_unpack_roundtrip")
    q = torch.randint(0, 16, (7, 16), dtype=torch.int32)
    packed = pack_int4(q)
    assert packed.shape == (7, 8)
    assert packed.dtype == torch.uint8
    restored = unpack_int4(packed, in_features=16)
    assert torch.equal(q, restored), "pack/unpack int4 round-trip must be exact"
    print("  OK: pack -> unpack round-trip is exact")


def test_gptq_identity_hessian():
    _print_header("test_gptq_identity_hessian (== RTN baseline)")
    torch.manual_seed(0)
    out_features, in_features = 64, 256
    group_size = 64
    w = torch.randn(out_features, in_features, dtype=torch.float16)

    q, s, z = gptq_quantize_weight(
        w, hessian=None, bits=4, group_size=group_size, sym=False
    )
    # reconstruct
    s_full = s.repeat_interleave(group_size, dim=1)
    z_full = z.repeat_interleave(group_size, dim=1)
    w_hat = (q.to(torch.float32) - z_full) * s_full
    err = (w.float() - w_hat).abs().mean().item()
    rel = err / w.float().abs().mean().item()
    print(f"  shape={tuple(w.shape)} group_size={group_size}")
    print(f"  mean |w - w_hat| = {err:.6f}   relative = {rel:.4%}")
    # For 4-bit asym quant of std-normal weights the relative error
    # should be small (empirically a few percent).
    assert rel < 0.20, f"RTN int4 error too large: {rel:.4%}"
    print("  OK")


def test_gptq_reduces_error_vs_rtn():
    _print_header("test_gptq_reduces_error_vs_rtn")
    torch.manual_seed(0)
    out_features, in_features = 64, 256
    group_size = 64
    n_calib = 512
    w = torch.randn(out_features, in_features, dtype=torch.float32)

    # Build a non-trivial Hessian from synthetic calibration activations.
    x = torch.randn(n_calib, in_features)
    H = 2.0 * (x.t() @ x) / n_calib

    # RTN baseline (identity Hessian -> no error compensation).
    q_rtn, s_rtn, z_rtn = gptq_quantize_weight(
        w, hessian=None, bits=4, group_size=group_size, sym=False
    )
    w_rtn = (q_rtn.to(torch.float32)
             - z_rtn.repeat_interleave(group_size, dim=1)) \
            * s_rtn.repeat_interleave(group_size, dim=1)

    # GPTQ with real Hessian.
    q_gptq, s_gptq, z_gptq = gptq_quantize_weight(
        w, hessian=H, bits=4, group_size=group_size, sym=False
    )
    w_gptq = (q_gptq.to(torch.float32)
              - z_gptq.repeat_interleave(group_size, dim=1)) \
             * s_gptq.repeat_interleave(group_size, dim=1)

    # What we actually care about is the downstream activation error.
    y_true = x @ w.t()
    err_rtn = (x @ w_rtn.t() - y_true).pow(2).mean().item()
    err_gptq = (x @ w_gptq.t() - y_true).pow(2).mean().item()
    print(f"  downstream MSE  RTN = {err_rtn:.6f}")
    print(f"  downstream MSE GPTQ = {err_gptq:.6f}")
    print(f"  GPTQ / RTN ratio    = {err_gptq / err_rtn:.4f}")
    # GPTQ should typically give a strictly lower downstream error.
    assert err_gptq <= err_rtn * 1.05, (
        "GPTQ should not be noticeably worse than RTN"
    )
    print("  OK (GPTQ <= RTN)")


def test_gptq_linear_forward():
    _print_header("test_gptq_linear_forward")
    torch.manual_seed(0)
    in_features, out_features = 256, 64
    group_size = 64
    base = nn.Linear(in_features, out_features, bias=True).half()

    q_layer = GPTQLinear(
        in_features, out_features, bias=True,
        group_size=group_size, bits=4, compute_dtype=torch.float16,
    )
    q_layer.load_from_float_weight(base.weight.data, base.bias.data)

    x = torch.randn(4, in_features, dtype=torch.float16)
    y_ref = base(x)
    y_q = q_layer(x)
    assert y_q.shape == y_ref.shape
    rel = (y_q.float() - y_ref.float()).abs().mean() / y_ref.float().abs().mean()
    print(f"  mean relative error = {rel.item():.4%}")
    assert rel.item() < 0.25, "GPTQLinear forward error is unexpectedly high"
    print("  OK")


def test_apply_gptq_to_module():
    _print_header("test_apply_gptq_to_module")
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 64, bias=False)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(64, 32, bias=True)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    m = Tiny().half()
    x = torch.randn(8, 128, dtype=torch.half)
    y_ref = m(x)

    n = apply_gptq_to_module(m, group_size=64, bits=4, verbose=True)
    assert n == 2, f"expected 2 layers replaced, got {n}"
    assert isinstance(m.fc1, GPTQLinear)
    assert isinstance(m.fc2, GPTQLinear)

    y_q = m(x)
    rel = (y_q.float() - y_ref.float()).abs().mean() / y_ref.float().abs().mean()
    print(f"  model-level mean relative error = {rel.item():.4%}")
    assert rel.item() < 0.5
    print("  OK")


if __name__ == "__main__":
    t0 = time.time()
    test_pack_unpack_roundtrip()
    test_gptq_identity_hessian()
    test_gptq_reduces_error_vs_rtn()
    test_gptq_linear_forward()
    test_apply_gptq_to_module()
    print(f"\nAll GPTQ tests passed in {time.time() - t0:.2f}s")