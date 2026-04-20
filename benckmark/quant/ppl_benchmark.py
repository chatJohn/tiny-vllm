"""Perplexity benchmark for GPTQ quantization on ``nanovllm``.

Goal
----
Measure and compare the perplexity (PPL) of a HuggingFace causal LM
*before* and *after* applying the in-repo GPTQ implementation
(``nanovllm.utils.gptq.apply_gptq_to_module``).

Why we load the model via HuggingFace here
------------------------------------------
The ``nanovllm`` engine is designed for generation and returns logits only
at the last position of each sequence, which is not suitable for
perplexity evaluation that needs *per-token* logits.  Since our GPTQ
quantizer operates on any ``nn.Linear``-like layer, we simply load the
model with ``transformers`` and apply the *same* ``apply_gptq_to_module``
function on it.  The quantization algorithm under evaluation is therefore
identical to the one used inside ``nanovllm``.

Usage
-----
    python benckmark/quant/ppl_benchmark.py \
        --model-path /data1/home/shxgou/code/models/qwen3-0.6b \
        --seq-len 1024 --stride 512

    # use your own evaluation text
    python benckmark/quant/ppl_benchmark.py \
        --model-path /path/to/model --text-file /path/to/corpus.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo root importable so we can reuse nanovllm.utils.gptq
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nanovllm.utils.gptq import apply_gptq_to_module  # noqa: E402


# ---------------------------------------------------------------------------
# Built-in evaluation corpus
# ---------------------------------------------------------------------------
# A small but non-trivial English corpus (~ wikitext-style).  We use it when
# the user doesn't pass ``--text-file`` or ``--hf-dataset`` so the benchmark
# works fully offline and with zero extra dependencies.
DEFAULT_CORPUS = """\
Artificial intelligence has become a defining technology of the early
twenty-first century.  Its rapid progress is driven mainly by large neural
networks trained on massive amounts of data.  Among these networks, the
Transformer architecture, introduced in 2017, has proved to be especially
effective for natural language understanding and generation.  A Transformer
relies on a mechanism called self-attention, which allows every token in a
sequence to attend to every other token.  This simple yet powerful idea has
scaled remarkably well, leading to models such as BERT, GPT and T5.

Large language models are typically trained in two stages.  First, the model
is pre-trained on a vast corpus of text with a self-supervised objective,
most commonly next-token prediction.  Then it is fine-tuned on smaller
task-specific datasets, or adapted via techniques such as reinforcement
learning from human feedback.  Pre-training imbues the model with broad
knowledge about the world and about language itself, while fine-tuning
sharpens its behaviour for particular applications.

A persistent challenge is that state-of-the-art language models are
expensive to run.  A single model may contain tens or hundreds of billions
of parameters, which must be fetched from memory for each forward pass.
Quantization is therefore an essential technique: by representing each
weight with fewer bits, we reduce both the memory footprint and the memory
bandwidth required during inference.  Popular weight-only quantization
schemes include round-to-nearest (RTN), GPTQ and AWQ, which all aim to
preserve accuracy while compressing the model down to four or even three
bits per weight.

GPTQ, in particular, frames post-training quantization as a layer-wise
optimization problem.  Given calibration activations, it builds a Hessian
matrix for each linear layer and quantizes the weights one column at a
time, propagating the induced error to the columns that remain.  This
error-feedback strategy makes GPTQ substantially more accurate than naive
rounding at the same bit width, which is why it has become a de-facto
baseline for evaluating new quantization algorithms.

Mathematics also plays a central role in modern science.  Prime numbers,
for example, are the building blocks of the integers: every natural number
greater than one can be written in essentially one way as a product of
primes.  Despite their simple definition, primes exhibit surprisingly rich
structure, and questions about their distribution have fascinated
mathematicians for centuries.  The prime number theorem, proved at the end
of the nineteenth century, states that the number of primes not exceeding
a large integer n is approximately n divided by the natural logarithm of n.

Beyond pure mathematics, the same ideas show up in computer science,
cryptography and physics.  Public-key cryptosystems such as RSA exploit the
apparent difficulty of factoring the product of two large primes.  In
quantum computing, Shor's algorithm demonstrates that a sufficiently large
quantum computer could factor such numbers efficiently, which would break
RSA and force a migration to post-quantum cryptographic schemes.  These
connections illustrate how deeply interwoven classical and modern ideas
have become.
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_eval_text(args) -> str:
    """Load the evaluation corpus according to CLI arguments."""
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[data] loaded {len(text):,} characters from {args.text_file}")
        return text
    if args.hf_dataset:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            print(
                "[data] 'datasets' is not installed; falling back to the "
                "built-in corpus"
            )
        else:
            name, cfg, split = args.hf_dataset.split(":")
            ds = load_dataset(name, cfg, split=split)
            text = "\n\n".join(ds["text"])
            print(f"[data] loaded {len(text):,} characters from {args.hf_dataset}")
            return text
    print(f"[data] using built-in corpus ({len(DEFAULT_CORPUS):,} characters)")
    return DEFAULT_CORPUS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_hf_model(
    model_path: str,
    dtype: torch.dtype,
    device: torch.device,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Perplexity evaluation (sliding-window, standard "stride" eval)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    seq_len: int,
    stride: int,
    device: torch.device,
    desc: str = "",
) -> tuple[float, int]:
    """Sliding-window perplexity evaluation.

    We slide a window of length ``seq_len`` over the tokenized corpus with
    step ``stride`` and, for every window, accumulate the sum of negative
    log-likelihoods on the *new* tokens that were not scored in the previous
    window.  This is the standard evaluation protocol used e.g. by the
    HuggingFace Transformers documentation.

    Returns
    -------
    ppl : float
        The perplexity.
    n_tokens : int
        Number of tokens that contributed to the NLL sum.
    """
    input_ids = input_ids.to(device)
    total_len = input_ids.size(1)
    nlls = []
    n_tokens = 0
    prev_end = 0
    t0 = time.time()
    step = 0

    while prev_end < total_len:
        begin = max(0, prev_end + stride - seq_len) if prev_end > 0 else 0
        end = min(begin + seq_len, total_len)
        # number of *new* tokens whose loss we want to count in this window
        trg_len = end - prev_end

        window = input_ids[:, begin:end]
        labels = window.clone()
        # mask out tokens that we have already scored in the previous window
        labels[:, : -trg_len] = -100

        outputs = model(window, labels=labels)
        # HF returns the *mean* loss over the un-masked tokens.  We rescale
        # to a sum so that we can aggregate across windows.
        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            # Happens when every label is -100 (should not occur here).
            prev_end = end
            continue

        # number of tokens contributing to the loss in this window (= trg_len
        # minus the first position, which is never predicted).
        num_contributing = max(trg_len - 1 if begin == 0 and prev_end == 0 else trg_len, 1)
        # HF averages over (labels != -100) positions in the shifted view,
        # which equals num_contributing for this window's logic.
        nll_sum = loss.float() * num_contributing
        nlls.append(nll_sum)
        n_tokens += num_contributing

        step += 1
        if step % 8 == 0 or end == total_len:
            elapsed = time.time() - t0
            print(
                f"  [{desc}] step {step:3d}  window {begin:6d}:{end:<6d}  "
                f"tokens={n_tokens:6d}  elapsed={elapsed:5.1f}s"
            )
        prev_end = end
        if end == total_len:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = float(torch.exp(total_nll / max(n_tokens, 1)))
    return ppl, n_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def model_param_bytes(model: nn.Module) -> int:
    """Total bytes used by ``model`` parameters **and** registered buffers.

    This counts the int8 buffers of ``GPTQLinear`` correctly because they
    are registered via ``register_buffer``.
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark: fp baseline vs. GPTQ int4"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/home/shxgou/code/models/qwen3-0.6b",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to a .txt file used as the evaluation corpus.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HF dataset spec 'name:config:split', e.g. "
             "'wikitext:wikitext-2-raw-v1:test'. Requires the 'datasets' package.",
    )
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Evaluate only the quantized model (useful for debugging).",
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Evaluate only the fp baseline.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    # ---------------- data ---------------------------------------------------
    text = load_eval_text(args)

    print(f"\n[config] model       = {args.model_path}")
    print(f"[config] device      = {device}  dtype={dtype}")
    print(f"[config] seq_len     = {args.seq_len}   stride = {args.stride}")
    print(f"[config] group_size  = {args.quant_group_size}   bits = {args.quant_bits}")

    # ---------------- tokenize ----------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    print(f"[tokenize] total tokens = {input_ids.numel():,}")

    # ---------------- baseline PPL ------------------------------------------
    ppl_fp = None
    fp_bytes = None
    if not args.skip_baseline:
        print("\n================ FP BASELINE ================")
        model_fp, _ = load_hf_model(args.model_path, dtype, device)
        fp_bytes = model_param_bytes(model_fp)
        print(f"[fp]  param+buffer bytes = {human_bytes(fp_bytes)}")
        t0 = time.time()
        ppl_fp, n_tok = evaluate_perplexity(
            model_fp, input_ids, args.seq_len, args.stride, device, desc="fp"
        )
        print(f"[fp]  PPL = {ppl_fp:.4f}   on {n_tok} tokens   "
              f"({time.time() - t0:.1f}s)")
        # release memory before quantized run
        del model_fp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------- GPTQ PPL ----------------------------------------------
    ppl_gptq = None
    gptq_bytes = None
    if not args.skip_quant:
        print("\n================ GPTQ INT4 ==================")
        model_q, _ = load_hf_model(args.model_path, dtype, device)
        # quantize every linear layer in the transformer blocks.
        # Skip the output head (``lm_head``) for safety, mirroring the
        # nanovllm-side behaviour which skips ParallelLMHead.
        t0 = time.time()
        head = getattr(model_q, "lm_head", None)
        replaced = apply_gptq_to_module(
            model_q.model if hasattr(model_q, "model") else model_q,
            group_size=args.quant_group_size,
            bits=args.quant_bits,
            compute_dtype=dtype,
            verbose=False,
        )
        print(f"[gptq] quantized {replaced} linear layers "
              f"({time.time() - t0:.1f}s)")
        if head is not None:
            print(f"[gptq] lm_head kept in {next(head.parameters()).dtype}")

        gptq_bytes = model_param_bytes(model_q)
        print(f"[gptq] param+buffer bytes = {human_bytes(gptq_bytes)}")

        t0 = time.time()
        ppl_gptq, n_tok = evaluate_perplexity(
            model_q, input_ids, args.seq_len, args.stride, device, desc="gptq"
        )
        print(f"[gptq] PPL = {ppl_gptq:.4f}   on {n_tok} tokens   "
              f"({time.time() - t0:.1f}s)")
        del model_q
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------- summary -----------------------------------------------
    print("\n=================== SUMMARY ===================")
    print(f"  model                : {args.model_path}")
    print(f"  corpus tokens        : {input_ids.numel():,}")
    print(f"  seq_len / stride     : {args.seq_len} / {args.stride}")
    if ppl_fp is not None:
        print(f"  FP {args.dtype:<9}        : PPL = {ppl_fp:8.4f}   "
              f"size = {human_bytes(fp_bytes)}")
    if ppl_gptq is not None:
        print(f"  GPTQ int{args.quant_bits} g{args.quant_group_size:<4}    "
              f": PPL = {ppl_gptq:8.4f}   size = {human_bytes(gptq_bytes)}")
    if ppl_fp is not None and ppl_gptq is not None:
        rel = (ppl_gptq - ppl_fp) / ppl_fp * 100.0
        comp = fp_bytes / gptq_bytes if gptq_bytes else 0.0
        print(f"  PPL delta            : {ppl_gptq - ppl_fp:+.4f}  "
              f"({rel:+.2f}%)")
        print(f"  compression ratio    : {comp:.2f}x")
    print("================================================")


if __name__ == "__main__":
    main()
