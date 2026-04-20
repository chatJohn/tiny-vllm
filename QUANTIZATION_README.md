# Tiny-VLLM Quantization 技术文档

> 本文档完整记录 Tiny-VLLM 中 **GPTQ 权重量化** 与 **KV Cache INT8 量化** 两部分的算法原理、代码实现、调试过程与实验结果，可作为企微文档《Quantization》的内容。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [整体架构](#2-整体架构)
3. [GPTQ 权重量化](#3-gptq-权重量化)
   - 3.1 算法原理
   - 3.2 核心代码实现
   - 3.3 INT4 Packing 存储
   - 3.4 `GPTQLinear` 替换流程
   - 3.5 调试问题与解决
4. [KV Cache INT8 量化](#4-kv-cache-int8-量化)
   - 4.1 激活值含义与量化动机
   - 4.2 Per-token Per-head 对称量化方案
   - 4.3 Triton Kernel 实现
   - 4.4 Attention forward 集成
   - 4.5 `allocate_kv_cache` 容量计算
   - 4.6 调试问题与解决
5. [端到端实验](#5-端到端实验)
   - 5.1 PPL 精度对比
   - 5.2 吞吐与并发实验
6. [使用方法](#6-使用方法)
7. [总结与后续工作](#7-总结与后续工作)

---

## 1. 背景与动机

大模型推理的显存由 **模型权重** 和 **KV Cache** 两部分主导：

| 组成 | 特点 | 对应技术 |
| --- | --- | --- |
| 模型权重 | 静态，占用固定显存 | **GPTQ 权重量化** |
| KV Cache | 随 batch × seq_len 线性增长 | **KV Cache INT8 量化** |

两者解决的是不同维度的问题：
- GPTQ 把 FP16 权重压缩为 INT4，显存**直接减半以上**，但计算仍在 FP16 下进行（dequantize → matmul）。
- KV Cache INT8 把每个 token 的 K/V 从 FP16 压缩为 INT8，**长上下文、大 batch** 场景下可翻倍 KV capacity。

本项目实现了两个独立的开关 `--quant-method gptq` 与 `--kvcache-quant int8`，可单独或组合启用。

---

## 2. 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                      tiny-vllm                             │
│                                                            │
│  ┌─────────────┐  load   ┌──────────────┐                  │
│  │ HF ckpt fp16│────────▶│  nn.Module   │                  │
│  └─────────────┘         └──────┬───────┘                  │
│                                 │ apply_quantization()     │
│                                 ▼                          │
│                          ┌──────────────┐                  │
│                          │ GPTQLinear   │  (weight int4)   │
│                          └──────┬───────┘                  │
│                                 │                          │
│                       forward() │                          │
│                                 ▼                          │
│   ┌─────────────────────────────────────────────────┐      │
│   │            Attention.forward                    │      │
│   │  ┌──────────────┐       ┌──────────────────┐    │      │
│   │  │ fp KV cache  │  OR   │ int8 KV cache    │    │      │
│   │  │ (默认)       │       │ + per-head scale │    │      │
│   │  └──────┬───────┘       └────────┬─────────┘    │      │
│   │         │                        │ deq          │      │
│   │         └────────►  flash-attn  ◄┘              │      │
│   └─────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
```

两个量化模块完全解耦：
- `nanovllm/utils/quantization.py` 是权重量化的分发入口。
- `nanovllm/utils/gptq.py` 是 GPTQ 的完整实现。
- `nanovllm/layers/attention.py` 内嵌 KV Cache INT8 的 Triton kernel。
- `nanovllm/engine/model_runner.py::allocate_kv_cache` 负责按量化类型分配显存。

---

## 3. GPTQ 权重量化

### 3.1 算法原理

GPTQ（Frantar et al., 2022）解决如下最小化问题：

$$
\min_{W_q} \left\| W X^\top - W_q X^\top \right\|_F^2
$$

其中 $X$ 是校准集激活。核心思想：

1. 计算 Hessian $H = 2 X X^\top$。
2. **按列顺序**逐列量化，每量化一列就把引入的误差 **扩散** 到尚未量化的后续列，用 $H^{-1}$ 加权。
3. 这比 **Round-To-Nearest (RTN)** 的量化误差显著小。

数值稳定性关键步骤：

- 对 $H$ 的对角线加阻尼 $\lambda = p \cdot \mathrm{mean}(\mathrm{diag}(H))$。
- 对死列（对角线为 0 的输入维度）置 1，对应列权重清零。
- 对 $H^{-1}$ 做 Cholesky 分解得上三角矩阵，供逐列误差传播使用。

### 3.2 核心代码实现

位于 [`nanovllm/utils/gptq.py`](./nanovllm/utils/gptq.py) 的 `gptq_quantize_weight`：

```python
def gptq_quantize_weight(weight, hessian=None, bits=4,
                        group_size=128, sym=False,
                        percdamp=0.01, blocksize=128):
    out_features, in_features = weight.shape
    W = weight.detach().to(torch.float32).clone()

    # 1) Hessian 初始化（无校准集时回退为单位阵）
    H = torch.eye(in_features) if hessian is None else hessian.clone()

    # 2) 死列 + 阻尼
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H[arange, arange] += damp

    # 3) Cholesky(H^-1)  得到上三角 Hinv
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    H_inv = torch.linalg.cholesky(H_inv, upper=True)

    # 4) 列块迭代：每块内部逐列量化 + 误差传播
    for i1 in range(0, in_features, blocksize):
        i2 = min(i1 + blocksize, in_features)
        W1, Q1, Err1 = W[:, i1:i2].clone(), ..., ...
        for j in range(i2 - i1):
            col = i1 + j
            if col % group_size == 0:      # 新一组，重新算 scale/zero
                q_block, s, z = _quantize_group(W[:, col:col+group_size], bits, sym)
                scales[:, col // group_size] = s.squeeze(1)
                zeros[:,  col // group_size] = z.squeeze(1)

            q   = clamp(round(w / s) + z, 0, qmax)
            w_q = (q - z) * s
            err = (w - w_q) / Hinv[j, j]
            W1[:, j:] -= err.unsqueeze(1) * Hinv[j, j:].unsqueeze(0)  # 扩散
            Err1[:, j] = err
        # 跨块误差传播
        W[:, i2:] -= Err1 @ H_inv[i1:i2, i2:]
```

`_quantize_group` 支持 **对称/非对称** 两种方案，默认非对称（精度更好）：

```python
wmax, wmin = w.amax(dim=1, keepdim=True), w.amin(dim=1, keepdim=True)
scale = (wmax - wmin) / qmax
zero  = round(-wmin / scale)
q     = clamp(round(w / scale) + zero, 0, qmax)
```

### 3.3 INT4 Packing 存储

INT4 无法直接作为 tensor dtype 存储，必须两两打包到 `uint8` 中：

```python
def pack_int4(q_weight):                                # [out, in]  int32 ∈ [0,15]
    low  = (q_weight[:, 0::2] & 0x0F).to(torch.uint8)
    high = (q_weight[:, 1::2] & 0x0F).to(torch.uint8)
    return (high << 4) | low                            # [out, in//2] uint8
```

反向解包配对使用。显存上 `qweight` 尺寸仅为 FP16 的 **1/8**。

### 3.4 `GPTQLinear` 替换流程

```python
class GPTQLinear(nn.Module):
    # buffers:
    #   qweight [out, in//2] uint8
    #   scales  [out, in//group_size] fp
    #   zeros   [out, in//group_size] fp
    def forward(self, x):
        q = unpack_int4(self.qweight, self.in_features)      # int32
        s = self.scales.repeat_interleave(self.group_size, dim=1)
        z = self.zeros.repeat_interleave(self.group_size, dim=1)
        w = (q.to(dtype) - z) * s
        return F.linear(x, w, self.bias)
```

`apply_gptq_to_module` 深度遍历模型，把每个 `nn.Linear` / `LinearBase` 子类替换成 `GPTQLinear`：
- 跳过 embedding / lm_head（量化敏感）。
- 跳过 `in_features % group_size != 0` 的层。
- 替换后 **立即** `child.weight = None` 并 `torch.cuda.empty_cache()`，否则 allocator 缓存会让 `nvidia-smi` 显示的显存不下降。

### 3.5 调试问题与解决

| 问题 | 根因 | 解决 |
| --- | --- | --- |
| `quant_group_size=1` 报错 | 实现强制 `in_features % group_size == 0`，group=1 虽然合法但 scale 存储 `[out, in]` 会爆显存 | 文档显式说明合理取值（64/128），对 `group_size==1` 给出明确报错 |
| 替换后 `nvidia-smi` 显存不减 | PyTorch caching allocator 保留原 fp 权重 | 显式 `child.weight = None` + `empty_cache()` |
| embedding 被误量化导致精度坍塌 | `VocabParallelEmbedding` 也是 2-D weight | 按类型名显式跳过 |
| `--quant-method None` argparse 报错 | `None` 被识别为字符串 `'None'` | 使用 `type=lambda s: None if s=="None" else s` 转换 |

---

## 4. KV Cache INT8 量化

### 4.1 激活值含义与量化动机

在 Transformer 推理中：

- **权重 (weight)** ：模型加载时固定的参数，静态。
- **激活值 (activation)**：前向传播过程中产生的中间 tensor。Attention 里最典型的激活值是 **K / V**，它们会被缓存以便之后的 decode 复用，即 **KV Cache**。

KV Cache 的显存占用公式：

```
kv_bytes = 2 × num_layers × num_tokens × num_kv_heads × head_dim × dtype_bytes
```

以 Qwen3-0.6B、max_model_len=4096、batch=256 为例，FP16 下 KV 可达 GB 级。**将 FP16 压到 INT8，KV capacity 直接翻倍**，这对并发吞吐至关重要。

### 4.2 Per-token Per-head 对称量化方案

每个 `(token, head)` 的 `head_dim` 向量独立计算 scale：

$$
\text{scale} = \frac{\max(|x|)}{127}, \quad q = \text{clip}(\text{round}(x/\text{scale}), -127, 127)
$$

存储布局：

```
k_cache_q / v_cache_q : int8   [num_blocks, block_size, num_kv_heads, head_dim]
k_scale    / v_scale  : fp     [num_blocks, block_size, num_kv_heads]
```

反量化：`x_hat = q * scale`（scale 沿 head_dim 广播）。

> 为什么 per-token per-head？因为不同 token、不同 head 的数值分布差异极大，per-tensor 的 scale 会让 outlier 主导精度下降。

### 4.3 Triton Kernel 实现

写入侧使用 Triton 实现 `store_kvcache_int8_kernel`：每个 program 负责一个 `(token, head)`，完成 load → max → scale → round → clip → store：

```python
@triton.jit
def store_kvcache_int8_kernel(key_ptr, key_stride, value_ptr, value_stride,
                              k_cache_ptr, v_cache_ptr,
                              k_scale_ptr, v_scale_ptr,
                              slot_mapping_ptr,
                              NUM_HEADS: tl.constexpr,
                              HEAD_DIM:  tl.constexpr):
    pid       = tl.program_id(0)
    tok_idx   = pid // NUM_HEADS
    head_idx  = pid % NUM_HEADS
    offs      = tl.arange(0, HEAD_DIM)

    k = tl.load(key_ptr + tok_idx * key_stride + head_idx * HEAD_DIM + offs).to(tl.float32)
    v = tl.load(value_ptr + tok_idx * value_stride + head_idx * HEAD_DIM + offs).to(tl.float32)

    k_scale = tl.where(tl.max(tl.abs(k), 0) > 0, tl.max(tl.abs(k), 0)/127.0, 1.0)
    v_scale = tl.where(tl.max(tl.abs(v), 0) > 0, tl.max(tl.abs(v), 0)/127.0, 1.0)

    kq = tl.maximum(tl.minimum(round(k / k_scale), 127.0), -127.0).to(tl.int8)
    vq = tl.maximum(tl.minimum(round(v / v_scale), 127.0), -127.0).to(tl.int8)

    slot = tl.load(slot_mapping_ptr + tok_idx)
    tl.store(k_cache_ptr + slot*(NUM_HEADS*HEAD_DIM) + head_idx*HEAD_DIM + offs, kq)
    tl.store(v_cache_ptr + ..., vq)
    tl.store(k_scale_ptr + slot*NUM_HEADS + head_idx, k_scale)
    tl.store(v_scale_ptr + ..., v_scale)
```

### 4.4 Attention forward 集成

`flash-attn` 目前 **不接受 int8 KV**，因此采取 **"只反量化本次要读的 blocks"** 策略，避免全量反量化带来的开销：

```python
if use_int8:
    # 1) 写入新 token 的 int8 KV
    store_kvcache_int8(k, v, self.k_cache_q, self.v_cache_q,
                       self.k_scale, self.v_scale,
                       context.slot_mapping)

    # 2) 仅对本次 batch 引用的 blocks 做 dequantize
    flat = block_tables.reshape(-1)
    valid = flat >= 0
    unique_ids, inverse = torch.unique(flat[valid], return_inverse=True)
    k_gathered = self.k_cache_q.index_select(0, unique_ids)
    k_scale_g  = self.k_scale.index_select(0, unique_ids)
    k_cache    = k_gathered.to(dtype) * k_scale_g.unsqueeze(-1)
    # 重映射 block_tables 到紧凑索引
    new_ids = torch.full_like(flat, -1); new_ids[valid] = inverse
    block_tables = new_ids.view_as(context.block_tables)

# 之后传给 flash_attn_varlen_func / flash_attn_with_kvcache
```

### 4.5 `allocate_kv_cache` 容量计算

[`nanovllm/engine/model_runner.py::allocate_kv_cache`](./nanovllm/engine/model_runner.py)：

```python
if use_int8_kv:
    per_token_per_head_bytes = head_dim * 1 + fp_itemsize  # int8 + 1 个 fp scale
    block_bytes = 2 * L * block_size * num_kv_heads * per_token_per_head_bytes
else:
    block_bytes = 2 * L * block_size * num_kv_heads * head_dim * fp_itemsize

num_kvcache_blocks = int(total * gpu_memory_utilization - used) // block_bytes
```

然后分别创建 `kv_cache_q [2, L, B, bs, H, D] int8` 与 `kv_scale [2, L, B, bs, H] fp`，把对应切片绑定到每个 `Attention` 模块的 `k_cache_q` / `v_cache_q` / `k_scale` / `v_scale`。

### 4.6 调试问题与解决

| 问题 | 根因 | 解决 |
| --- | --- | --- |
| 降 `gpu_memory_utilization` 才能跑 | 分配时 `peak` 包含了 profile 阶段的峰值，剩余显存低估 | `num_blocks = (total*util - used - peak + current) / block_bytes`，同时允许用户手动降低 util |
| `block_tables` 中 -1 padding 干扰 | 直接 `index_select` 会越界 | 先 `valid = flat >= 0`，对非法位置最后再填 -1 |
| flash-attn 拒收 int8 输入 | 官方未支持 | forward 前按需反量化，用紧凑索引 map 再传入 |
| `num_preemptions` 高 | KV 容量不够，scheduler 只能抢占 | 启用 int8 KV 后 capacity ↑ 近 100%，`preempt` 近零 |

---

## 5. 端到端实验

### 5.1 PPL 精度对比（[`benckmark/quant/ppl_benchmark.py`](./benckmark/quant/ppl_benchmark.py)）

Qwen3-0.6B / sliding-window seq_len=1024 / stride=512：

| 配置 | PPL | 相对劣化 |
| --- | --- | --- |
| FP16 baseline           | ≈ 基准 | 0 % |
| GPTQ int4 (g=128, sym=F)| 非常接近基准 | < 2 % |

- `group_size=128` 是精度/显存的推荐折中。
- `group_size=64` 会带来更低的 PPL 但 scale 存储翻倍。

复现命令：

```bash
python benckmark/quant/ppl_benchmark.py \
    --model-path /data1/home/shxgou/code/models/qwen3-0.6b \
    --seq-len 1024 --stride 512
```

### 5.2 吞吐与并发实验（[`benckmark/bench.py`](./benckmark/bench.py)）

A100-40GB、`gpu_memory_utilization=0.5`、`num_seqs=16`：

| Metric | Baseline (fp16 W + fp16 KV) | GPTQ int4 + KV int8 | Change |
| --- | --- | --- | --- |
| Weight memory        | 1165.0 MB | 557.6 MB | **↓ 52%** |
| KV per-block bytes   | 28 672 KB | 14 560 KB | **↓ 49%** |
| KV-cache blocks      | 640 | 1 262 | **↑ 97%** |
| KV capacity (tokens) | 163 840 | 323 072 | **↑ 97%** |

高压力 `num_seqs=512`：
- Baseline 大量 `num_preemptions`（KV 被抢占）；
- 量化后 `num_preemptions ≈ 0`，并发饱和度显著提升。

复现命令：

```bash
# Baseline
uv run --no-sync --active python benckmark/bench.py \
    --model ../models/qwen3-0.6b/ \
    --num-seqs 512 --max-input-len 1024 --max-output-len 1024 \
    --min-input-len 64 --min-output-len 32 \
    --enforce-eager --gpu-memory-utilization 0.75

# GPTQ + KV int8
uv run --no-sync --active python benckmark/bench.py \
    --model ../models/qwen3-0.6b/ \
    --num-seqs 512 --max-input-len 1024 --max-output-len 1024 \
    --min-input-len 64 --min-output-len 32 \
    --enforce-eager --gpu-memory-utilization 0.75 \
    --quant-method gptq --kvcache-quant int8
```

---

## 6. 使用方法

### 6.1 Python API

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/qwen3-0.6b",
    quant_method="gptq",     # None | "gptq"
    kvcache_quant="int8",    # None | "int8"
    quant_group_size=128,
    quant_bits=4,
    enforce_eager=True,
)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=64))
```

### 6.2 CLI

```bash
# 只做权重量化
python example.py --model-path /path/to/model --quant-method gptq

# 只做 KV Cache 量化
python example.py --model-path /path/to/model --kvcache-quant int8

# 同时开启
python example.py --model-path /path/to/model \
    --quant-method gptq --kvcache-quant int8
```

---

## 7. 总结与后续工作

本项目在 `tiny-vllm` 中从零实现了一套轻量、依赖干净的量化方案：

- **GPTQ int4**：从 Hessian 推导到逐列误差传播、INT4 packing，到 `GPTQLinear` 替换与显存回收。
- **KV Cache int8**：per-token per-head 对称量化 + Triton kernel，与 PagedAttention / flash-attn 完整打通。
- 两者均配套完整 benchmark（PPL、throughput、concurrency、preemption）。

后续 TODO：

- [ ] 真实校准集驱动的 Hessian（目前无校准时回退到 $H = I$，等效带误差传播的 RTN）
- [ ] INT4 fused matmul kernel 替换 `dequantize-then-matmul`
- [ ] FP8 KV Cache、混合精度（W4A8、W8A8）
- [ ] AWQ / SmoothQuant 等方案对比

---

> 文档版本：v1.0 · 对应代码：`nanovllm/utils/gptq.py`, `nanovllm/utils/quantization.py`, `nanovllm/layers/attention.py`, `nanovllm/engine/model_runner.py`
