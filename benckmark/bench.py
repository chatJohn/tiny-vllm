"""Generic throughput benchmark for nanovllm (tiny-vllm).

Example usage::

    python benckmark/bench.py \
        --model ~/huggingface/Qwen3-0.6B/ \
        --num-seqs 256 \
        --max-input-len 1024 \
        --max-output-len 1024

    # With GPTQ int4 weights + int8 KV-cache
    python benckmark/bench.py --model ~/huggingface/Qwen3-0.6B-GPTQ/ \
        --quant-method gptq --kvcache-quant int8

The script measures end-to-end throughput (tokens / second) over a batch of
randomly generated prompt token ids, mirroring the workload shape from the
original reference benchmark but exposed through CLI flags so it can be reused
for any model path / configuration.

In addition to raw throughput, it also reports *capacity / concurrency*
metrics that make the benefit of weight / KV-cache quantization visible:

- ``weight_bytes`` / ``weight_MB`` : resident GPU memory for model weights.
- ``kv_block_bytes`` / ``num_kvcache_blocks`` : paged-attention capacity.
- ``kv_total_MB`` : total GPU memory actually reserved for KV cache.
- ``kv_capacity_tokens`` : #tokens the engine can cache at the same time.
- ``avg_running_batch`` / ``max_running_batch`` : actual concurrency seen at
  runtime (how many sequences decode together in a step).
- ``avg_waiting`` / ``max_waiting`` : queue pressure.
- ``num_preemptions`` : how many times the scheduler had to evict a running
  sequence because KV blocks ran out -- quantization typically reduces this
  to zero.
"""

import argparse
import json
import os
import threading
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic throughput benchmark for tiny-vllm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / engine configuration
    parser.add_argument(
        "--model",
        type=str,
        default="~/huggingface/Qwen3-0.6B/",
        help="Path to the HuggingFace-format model directory.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (prompt + generated).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="Maximum number of concurrent sequences in the scheduler.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens processed per scheduler step.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory the engine is allowed to use.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel world size.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture (useful for debugging).",
    )
    parser.add_argument(
        "--kvcache-block-size",
        type=int,
        default=256,
        help="Paged-attention KV cache block size (must be multiple of 256).",
    )

    # Quantization configuration
    parser.add_argument(
        "--quant-method",
        type=str,
        default=None,
        choices=[None, "gptq"],
        help="Weight quantization method applied after load.",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=4,
        help="Quantization bit-width (only meaningful with --quant-method).",
    )
    parser.add_argument(
        "--quant-group-size",
        type=int,
        default=128,
        help="Quantization group size (only meaningful with --quant-method).",
    )
    parser.add_argument(
        "--kvcache-quant",
        type=str,
        default=None,
        choices=[None, "int8"],
        help="KV-cache quantization mode.",
    )

    # Workload configuration
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=256,
        help="Number of sequences to generate in the benchmark batch.",
    )
    parser.add_argument(
        "--min-input-len",
        type=int,
        default=100,
        help="Minimum random prompt length (tokens).",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=1024,
        help="Maximum random prompt length (tokens).",
    )
    parser.add_argument(
        "--min-output-len",
        type=int,
        default=100,
        help="Minimum random output length (tokens).",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Maximum random output length (tokens).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Upper bound (exclusive-ish) for the random prompt token ids.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=True,
        help="Ignore EOS so every sequence generates exactly max_tokens.",
    )
    parser.add_argument(
        "--no-ignore-eos",
        dest="ignore_eos",
        action="store_false",
        help="Honour EOS tokens (output length may be shorter than max_tokens).",
    )
    parser.add_argument(
        "--fixed-output-len",
        action="store_true",
        help="Use max_output_len for every sequence instead of a random length.",
    )

    # Benchmark meta configuration
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=True,
        help="Run a small warmup generation before the timed run.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip warmup.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="Number of timed iterations to run; reports average of all runs.",
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Enable tqdm progress bar during the timed generation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a JSON summary of the results to this path.",
    )

    return parser.parse_args()


def build_workload(args: argparse.Namespace):
    """Create random prompt token ids + sampling params for the benchmark."""
    min_in = max(1, args.min_input_len)
    max_in = max(min_in, args.max_input_len)
    min_out = max(1, args.min_output_len)
    max_out = max(min_out, args.max_output_len)

    prompt_token_ids = [
        [randint(0, args.vocab_size) for _ in range(randint(min_in, max_in))]
        for _ in range(args.num_seqs)
    ]

    if args.fixed_output_len:
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
                max_tokens=max_out,
            )
            for _ in range(args.num_seqs)
        ]
    else:
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
                max_tokens=randint(min_out, max_out),
            )
            for _ in range(args.num_seqs)
        ]

    return prompt_token_ids, sampling_params


def build_llm(args: argparse.Namespace) -> LLM:
    model_path = os.path.expanduser(args.model)
    llm_kwargs = dict(
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        kvcache_block_size=args.kvcache_block_size,
    )
    if args.quant_method is not None:
        llm_kwargs.update(
            quant_method=args.quant_method,
            quant_bits=args.quant_bits,
            quant_group_size=args.quant_group_size,
        )
    if args.kvcache_quant is not None:
        llm_kwargs["kvcache_quant"] = args.kvcache_quant
    return LLM(model_path, **llm_kwargs)


# ---------------------------------------------------------------------------
#  Capacity / concurrency metrics
# ---------------------------------------------------------------------------
def _compute_kv_block_bytes(llm: LLM) -> int:
    """Reproduce the per-block KV memory formula used by ``ModelRunner``.

    We don't have a direct ``block_bytes`` attribute, but we have every input
    it depends on on the engine config, so we just recompute it for reporting.
    """
    cfg = llm.model_runner.config
    hf = cfg.hf_config
    num_kv_heads = hf.num_key_value_heads // cfg.tensor_parallel_size
    head_dim = (
        hf.head_dim
        if hasattr(hf, "head_dim")
        else hf.hidden_size // hf.num_attention_heads
    )
    fp_itemsize = hf.torch_dtype.itemsize
    block_size = cfg.kvcache_block_size
    num_layers = hf.num_hidden_layers
    if cfg.kvcache_quant == "int8":
        per_token_per_head = head_dim * 1 + fp_itemsize  # int8 + fp scale
    else:
        per_token_per_head = head_dim * fp_itemsize
    return 2 * num_layers * block_size * num_kv_heads * per_token_per_head


def collect_capacity_metrics(llm: LLM) -> dict:
    """Collect static engine capacity metrics right after model loading.

    These are the numbers that make the memory benefit of quantization visible
    *before* a single request is generated:

    - resident weight memory (drops with GPTQ)
    - #KV-cache blocks, per-block bytes, total KV memory
    - how many tokens the engine can cache simultaneously
    """
    cfg = llm.model_runner.config
    num_blocks = cfg.num_kvcache_blocks
    block_size = cfg.kvcache_block_size
    block_bytes = _compute_kv_block_bytes(llm)
    kv_total_bytes = num_blocks * block_bytes
    kv_capacity_tokens = num_blocks * block_size

    # Weight memory is measured by ModelRunner right after load.  The model
    # runner prints it but doesn't expose it, so we re-query the allocator:
    # at this point only weights + KV cache are allocated, so
    # `allocated - kv_total_bytes` is a good estimate of weight bytes.
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    weight_bytes = max(0, allocated - kv_total_bytes)

    return {
        "weight_bytes": int(weight_bytes),
        "weight_MB": weight_bytes / 1024 ** 2,
        "gpu_allocated_bytes": int(allocated),
        "gpu_allocated_MB": allocated / 1024 ** 2,
        "gpu_reserved_bytes": int(reserved),
        "gpu_reserved_MB": reserved / 1024 ** 2,
        "num_kvcache_blocks": int(num_blocks),
        "kvcache_block_size": int(block_size),
        "kv_block_bytes": int(block_bytes),
        "kv_total_bytes": int(kv_total_bytes),
        "kv_total_MB": kv_total_bytes / 1024 ** 2,
        "kv_capacity_tokens": int(kv_capacity_tokens),
    }


class ConcurrencySampler:
    """Collect runtime concurrency metrics from the scheduler.

    Implementation detail: we *monkey-patch* ``scheduler.preempt`` to count
    evictions and spin a lightweight background thread that samples the
    ``running`` / ``waiting`` deque lengths and the KV block usage every
    ``interval`` seconds.  This adds no hooks to the engine source code.
    """

    def __init__(self, llm: LLM, interval: float = 0.05):
        self.llm = llm
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.running_samples: list[int] = []
        self.waiting_samples: list[int] = []
        self.block_used_samples: list[int] = []
        self.preempt_count = 0

        scheduler = llm.scheduler
        self._total_blocks = scheduler.block_manager.blocks.__len__()

        # Hook preempt so we can count evictions.
        self._orig_preempt = scheduler.preempt
        parent = self

        def _patched_preempt(seq):
            parent.preempt_count += 1
            return parent._orig_preempt(seq)

        scheduler.preempt = _patched_preempt

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Restore scheduler.preempt
        self.llm.scheduler.preempt = self._orig_preempt
        return False

    def _sample_loop(self):
        scheduler = self.llm.scheduler
        bm = scheduler.block_manager
        while not self._stop.is_set():
            # Deque length / set size reads are O(1) and thread-safe enough
            # for a monitoring purpose -- we tolerate a torn read here.
            try:
                self.running_samples.append(len(scheduler.running))
                self.waiting_samples.append(len(scheduler.waiting))
                self.block_used_samples.append(len(bm.used_block_ids))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def _max(xs):
            return max(xs) if xs else 0

        block_used_peak = _max(self.block_used_samples)
        block_used_avg = _avg(self.block_used_samples)
        return {
            "num_samples": len(self.running_samples),
            "avg_running_batch": _avg(self.running_samples),
            "max_running_batch": _max(self.running_samples),
            "avg_waiting": _avg(self.waiting_samples),
            "max_waiting": _max(self.waiting_samples),
            "avg_blocks_used": block_used_avg,
            "peak_blocks_used": block_used_peak,
            "total_blocks": int(self._total_blocks),
            "peak_block_utilization": (
                block_used_peak / self._total_blocks if self._total_blocks else 0.0
            ),
            "num_preemptions": int(self.preempt_count),
        }


def run_once(llm: LLM, prompt_token_ids, sampling_params, use_tqdm: bool):
    torch_sync()
    t0 = time.perf_counter()
    with ConcurrencySampler(llm) as sampler:
        outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=use_tqdm)
        torch_sync()
        elapsed = time.perf_counter() - t0
    concurrency = sampler.summary()

    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    # Count the tokens actually produced (respects ignore_eos=False etc.)
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    return {
        "elapsed_s": elapsed,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "concurrency": concurrency,
    }


def torch_sync() -> None:
    """Synchronize CUDA (if available) to get accurate timing."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def format_summary(args, per_iter_stats, capacity):
    n = len(per_iter_stats)
    avg_elapsed = sum(s["elapsed_s"] for s in per_iter_stats) / n
    avg_input = sum(s["input_tokens"] for s in per_iter_stats) / n
    avg_output = sum(s["output_tokens"] for s in per_iter_stats) / n
    avg_total = sum(s["total_tokens"] for s in per_iter_stats) / n

    output_tput = avg_output / avg_elapsed
    total_tput = avg_total / avg_elapsed
    per_req_latency = avg_elapsed / args.num_seqs

    # Aggregate runtime concurrency metrics across iterations
    def _avg(key):
        return sum(s["concurrency"][key] for s in per_iter_stats) / n

    def _max(key):
        return max(s["concurrency"][key] for s in per_iter_stats)

    def _sum(key):
        return sum(s["concurrency"][key] for s in per_iter_stats)

    concurrency_avg = {
        "avg_running_batch": _avg("avg_running_batch"),
        "max_running_batch": _max("max_running_batch"),
        "avg_waiting": _avg("avg_waiting"),
        "max_waiting": _max("max_waiting"),
        "avg_blocks_used": _avg("avg_blocks_used"),
        "peak_blocks_used": _max("peak_blocks_used"),
        "total_blocks": per_iter_stats[0]["concurrency"]["total_blocks"],
        "peak_block_utilization": _max("peak_block_utilization"),
        "num_preemptions": int(_sum("num_preemptions")),
    }

    summary = {
        "model": os.path.expanduser(args.model),
        "num_seqs": args.num_seqs,
        "iterations": n,
        "avg_elapsed_s": avg_elapsed,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "avg_total_tokens": avg_total,
        "output_throughput_tok_s": output_tput,
        "total_throughput_tok_s": total_tput,
        "per_request_latency_s": per_req_latency,
        "quant_method": args.quant_method,
        "kvcache_quant": args.kvcache_quant,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "capacity": capacity,
        "concurrency": concurrency_avg,
        "per_iter": per_iter_stats,
    }
    return summary


def print_summary(summary: dict) -> None:
    cap = summary["capacity"]
    conc = summary["concurrency"]
    print()
    print("=" * 72)
    print(" Benchmark summary")
    print("=" * 72)
    print(f"  Model                 : {summary['model']}")
    print(f"  # sequences           : {summary['num_seqs']}")
    print(f"  # iterations          : {summary['iterations']}")
    print(f"  Tensor parallel size  : {summary['tensor_parallel_size']}")
    print(f"  enforce_eager         : {summary['enforce_eager']}")
    print(f"  max_model_len         : {summary['max_model_len']}")
    print(f"  quant_method          : {summary['quant_method']}")
    print(f"  kvcache_quant         : {summary['kvcache_quant']}")
    print("-" * 72)
    print(" Capacity / memory (static, after model load)")
    print("-" * 72)
    print(f"  Weight memory         : {cap['weight_MB']:.1f} MB")
    print(f"  GPU allocated / resvd : {cap['gpu_allocated_MB']:.1f} / {cap['gpu_reserved_MB']:.1f} MB")
    print(f"  KV-cache blocks       : {cap['num_kvcache_blocks']} x {cap['kvcache_block_size']} tok")
    print(f"  KV per-block bytes    : {cap['kv_block_bytes'] / 1024:.1f} KB")
    print(f"  KV total memory       : {cap['kv_total_MB']:.1f} MB")
    print(f"  KV capacity (tokens)  : {cap['kv_capacity_tokens']}")
    print("-" * 72)
    print(" Concurrency (runtime, sampled during generation)")
    print("-" * 72)
    print(
        f"  Running batch         : avg={conc['avg_running_batch']:.1f}  "
        f"max={conc['max_running_batch']}"
    )
    print(
        f"  Waiting queue         : avg={conc['avg_waiting']:.1f}  "
        f"max={conc['max_waiting']}"
    )
    print(
        f"  KV blocks in use      : avg={conc['avg_blocks_used']:.1f}  "
        f"peak={conc['peak_blocks_used']}/{conc['total_blocks']}  "
        f"({conc['peak_block_utilization'] * 100:.1f}%)"
    )
    print(f"  Preemptions           : {conc['num_preemptions']}")
    print("-" * 72)
    print(f"  Avg elapsed           : {summary['avg_elapsed_s']:.2f} s")
    print(f"  Avg input  tokens     : {summary['avg_input_tokens']:.0f}")
    print(f"  Avg output tokens     : {summary['avg_output_tokens']:.0f}")
    print(f"  Avg total  tokens     : {summary['avg_total_tokens']:.0f}")
    print(f"  Output throughput     : {summary['output_throughput_tok_s']:.2f} tok/s")
    print(f"  Total  throughput     : {summary['total_throughput_tok_s']:.2f} tok/s")
    print(f"  Per-request latency   : {summary['per_request_latency_s']:.3f} s")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    seed(args.seed)

    print(f"[bench] loading model from {os.path.expanduser(args.model)} ...")
    llm = build_llm(args)

    capacity = collect_capacity_metrics(llm)
    print(
        f"[bench] capacity: weight={capacity['weight_MB']:.1f}MB  "
        f"kv_blocks={capacity['num_kvcache_blocks']}  "
        f"kv_capacity_tokens={capacity['kv_capacity_tokens']}  "
        f"kv_total={capacity['kv_total_MB']:.1f}MB"
    )

    if args.warmup:
        print("[bench] warmup ...")
        llm.generate(["Benchmark: "], SamplingParams(max_tokens=16), use_tqdm=False)

    per_iter_stats = []
    for i in range(args.num_iters):
        # Re-seed per iteration so the same workload is reproducible across runs
        seed(args.seed + i)
        prompt_token_ids, sampling_params = build_workload(args)
        print(
            f"[bench] iter {i + 1}/{args.num_iters}: "
            f"{args.num_seqs} seqs, "
            f"input~[{args.min_input_len},{args.max_input_len}], "
            f"output~[{args.min_output_len},{args.max_output_len}]"
        )
        stats = run_once(llm, prompt_token_ids, sampling_params, args.use_tqdm)
        print(
            f"[bench] iter {i + 1} done: "
            f"{stats['output_tokens']} out-tok in {stats['elapsed_s']:.2f}s "
            f"-> {stats['output_tokens'] / stats['elapsed_s']:.2f} tok/s"
        )
        per_iter_stats.append(stats)

    summary = format_summary(args, per_iter_stats, capacity)
    print_summary(summary)

    if args.output_json:
        out_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[bench] wrote JSON summary to {out_path}")


if __name__ == "__main__":
    main()
