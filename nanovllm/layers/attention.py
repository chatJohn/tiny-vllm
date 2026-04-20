import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn

from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


# ---------------------------------------------------------------------------
# INT8 KV-cache helpers
# ---------------------------------------------------------------------------
# We quantize the K/V of every newly produced token with a *per-token
# per-head* symmetric scheme:
#
#     scale = amax(|x|, over head_dim) / 127
#     q     = clip(round(x / scale), -127, 127)   # int8
#     x_hat = q * scale                           # used at read time
#
# Storage layout in the cache buffers
#   k_cache_q / v_cache_q : int8   [num_blocks, block_size, num_kv_heads, head_dim]
#   k_scale   / v_scale   : fp/bf  [num_blocks, block_size, num_kv_heads]
#
# Because flash-attn does not accept int8 caches, the Attention forward
# dequantizes the (logically) used cache back to the compute dtype before
# handing it to flash-attn.  The saving is therefore in **memory** only, not
# in compute -- which is exactly what we want for long contexts / large
# batches where the KV cache is the dominant consumer of GPU memory.
# ---------------------------------------------------------------------------
@triton.jit
def store_kvcache_int8_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,      # int8 [total_slots, num_kv_heads, head_dim]
    v_cache_ptr,      # int8
    k_scale_ptr,      # fp   [total_slots, num_kv_heads]
    v_scale_ptr,      # fp
    slot_mapping_ptr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # One program per (token, head).  Each program quantizes one head_dim
    # vector and writes one scalar scale.
    pid = tl.program_id(0)
    tok_idx = pid // NUM_HEADS
    head_idx = pid % NUM_HEADS

    # --- load the fp key / value vector --------------------------------------
    base_k = tok_idx * key_stride + head_idx * HEAD_DIM
    base_v = tok_idx * value_stride + head_idx * HEAD_DIM
    offs = tl.arange(0, HEAD_DIM)
    k = tl.load(key_ptr + base_k + offs).to(tl.float32)
    v = tl.load(value_ptr + base_v + offs).to(tl.float32)

    # --- compute per-head scales --------------------------------------------
    k_absmax = tl.max(tl.abs(k), axis=0)
    v_absmax = tl.max(tl.abs(v), axis=0)
    # Avoid division by zero when a head is (nearly) all-zero.
    k_scale = tl.where(k_absmax > 0.0, k_absmax / 127.0, 1.0)
    v_scale = tl.where(v_absmax > 0.0, v_absmax / 127.0, 1.0)

    # --- quantize & write out ------------------------------------------------
    kq = k / k_scale
    vq = v / v_scale
    # round-to-nearest, clip to int8 range
    kq = tl.extra.cuda.libdevice.round(kq)
    vq = tl.extra.cuda.libdevice.round(vq)
    kq = tl.maximum(tl.minimum(kq, 127.0), -127.0).to(tl.int8)
    vq = tl.maximum(tl.minimum(vq, 127.0), -127.0).to(tl.int8)

    slot = tl.load(slot_mapping_ptr + tok_idx)
    out_base = slot * (NUM_HEADS * HEAD_DIM) + head_idx * HEAD_DIM
    tl.store(k_cache_ptr + out_base + offs, kq)
    tl.store(v_cache_ptr + out_base + offs, vq)

    scale_offset = slot * NUM_HEADS + head_idx
    tl.store(k_scale_ptr + scale_offset, k_scale)
    tl.store(v_scale_ptr + scale_offset, v_scale)


def store_kvcache_int8(
    key: torch.Tensor,         # fp  [N, num_heads, head_dim]
    value: torch.Tensor,       # fp  [N, num_heads, head_dim]
    k_cache: torch.Tensor,     # int8 flattened over slots
    v_cache: torch.Tensor,     # int8
    k_scale: torch.Tensor,     # fp  flattened over slots
    v_scale: torch.Tensor,     # fp
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    grid = (N * num_heads,)
    store_kvcache_int8_kernel[grid](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        k_scale, v_scale,
        slot_mapping,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
    )


def dequantize_kv_cache(
    q_cache: torch.Tensor,          # int8, any shape ending with head_dim
    scale: torch.Tensor,            # fp,   same shape w/o the head_dim axis
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an int8 KV-cache buffer back to ``out_dtype``.

    We simply broadcast ``scale`` along ``head_dim`` and multiply.  This
    allocates a fresh tensor the size of the fp16 cache, so callers should
    invoke it lazily (e.g. once per forward pass).
    """
    # Broadcast scale to the last dim.
    return q_cache.to(out_dtype) * scale.unsqueeze(-1).to(out_dtype)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # fp16/bf16 caches (used when KV-cache quantization is off).
        self.k_cache = self.v_cache = torch.tensor([])
        # int8 caches + per-token per-head fp scales (used when on).
        # They are populated by ``ModelRunner.allocate_kv_cache`` when
        # ``config.kvcache_quant == "int8"``.
        self.k_cache_q: torch.Tensor | None = None
        self.v_cache_q: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None
        self.kv_quant: str | None = None  # None | "int8"

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()

        use_int8 = self.kv_quant == "int8" and self.k_cache_q is not None
        block_tables = context.block_tables

        if use_int8:
            # ---- write newly computed k/v into the int8 cache -----------
            if context.slot_mapping is not None and context.slot_mapping.numel():
                store_kvcache_int8(
                    k, v,
                    self.k_cache_q, self.v_cache_q,
                    self.k_scale, self.v_scale,
                    context.slot_mapping,
                )
            # ---- dequantize ONLY the blocks referenced by block_tables --
            # Reason: the int8 cache is usually allocated across ~all free
            # GPU memory.  Dequantizing the whole buffer every forward
            # pass would dominate runtime and defeat the purpose of
            # saving memory.  We therefore gather only the blocks that
            # will actually be read and build a compact fp cache.
            if block_tables is not None:
                # block_tables : [bs, max_num_blocks] (int32, -1 padded)
                flat = block_tables.reshape(-1)
                valid = flat >= 0
                valid_ids = flat[valid]
                # Unique, so that the same physical block is dequantized once
                unique_ids, inverse = torch.unique(
                    valid_ids, sorted=False, return_inverse=True
                )
                k_gathered = self.k_cache_q.index_select(0, unique_ids)
                v_gathered = self.v_cache_q.index_select(0, unique_ids)
                k_scale_g = self.k_scale.index_select(0, unique_ids)
                v_scale_g = self.v_scale.index_select(0, unique_ids)
                k_cache = dequantize_kv_cache(k_gathered, k_scale_g, q.dtype)
                v_cache = dequantize_kv_cache(v_gathered, v_scale_g, q.dtype)
                # Remap block_tables to the compact index space
                new_ids = torch.full_like(flat, -1)
                new_ids[valid] = inverse.to(new_ids.dtype)
                block_tables = new_ids.view_as(context.block_tables)
            else:
                # No paged cache in play (pure prefill without prefix cache).
                k_cache = dequantize_kv_cache(
                    self.k_cache_q, self.k_scale, q.dtype
                )
                v_cache = dequantize_kv_cache(
                    self.v_cache_q, self.v_scale, q.dtype
                )
        else:
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=block_tables,
            )
        else:  # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
