import argparse
import os

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main(args):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    quant_method = args.quant_method
    if quant_method == "none":
        quant_method = None
    kvcache_quant = args.kvcache_quant
    if kvcache_quant == "none":
        kvcache_quant = None
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        quant_method=quant_method,
        quant_group_size=args.quant_group_size,
        quant_bits=args.quant_bits,
        kvcache_quant=kvcache_quant,
        gpu_memory_utilization=args.gpu_memory_util,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm")
    argparse.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3MOE-30B-A3B"
    )
    argparse.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    argparse.add_argument("--enforce-eager", type=bool, default=True)
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--max-tokens", type=int, default=256)
    argparse.add_argument(
        "--quant-method",
        type=str,
        default="none",
        choices=["none", "gptq"],
        help="Post-training quantization method: 'none' keeps fp weights, "
             "'gptq' applies GPTQ INT4 quantization.",
    )
    argparse.add_argument(
        "--quant-group-size",
        type=int,
        default=128,
        help="Group size used by the quantizer along the in-features dim.",
    )
    argparse.add_argument(
        "--quant-bits",
        type=int,
        default=4,
        help="Number of bits (only 4 is currently supported for GPTQ).",
    )
    argparse.add_argument(
        "--kvcache-quant",
        type=str,
        default="none",
        choices=["none", "int8"],
        help="KV-cache quantization: 'none' keeps fp cache, 'int8' stores "
             "K/V as int8 with per-token per-head scales (~2x memory saving).",
    )
    argparse.add_argument(
        "--gpu-memory-util",
        type=float,
        default=0.9,
        help="Fraction of GPU memory the engine is allowed to use. "
             "Lower this (e.g. 0.2) when you want to see the weight-only "
             "memory footprint of quantization in nvidia-smi.",
    )
    args = argparse.parse_args()

    main(args)
