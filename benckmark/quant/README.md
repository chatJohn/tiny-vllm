# Quant Benchmarks

Benchmarks that measure the quality / speed / memory impact of the
quantization algorithms implemented in ``nanovllm``.

## Perplexity (PPL)

Compare the perplexity of a HuggingFace causal LM before and after applying
the in-repo GPTQ implementation (``nanovllm.utils.gptq``).

```bash
# Default: evaluate Qwen3-0.6B on the built-in mini corpus
python benckmark/quant/ppl_benchmark.py \
    --model-path /data1/home/shxgou/code/models/qwen3-0.6b

# Longer sequences (standard wikitext-style setup: 2048 / 1024)
python benckmark/quant/ppl_benchmark.py \
    --model-path /data1/home/shxgou/code/models/qwen3-0.6b \
    --seq-len 2048 --stride 1024

# Use your own evaluation text
python benckmark/quant/ppl_benchmark.py \
    --model-path /path/to/model --text-file /path/to/corpus.txt

# Use a HuggingFace dataset (requires `pip install datasets`)
python benckmark/quant/ppl_benchmark.py \
    --model-path /path/to/model \
    --hf-dataset wikitext:wikitext-2-raw-v1:test
```

### CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--model-path`         | `Qwen3-0.6B` | HF model directory to evaluate. |
| `--seq-len`            | 1024         | Sliding-window length used for PPL evaluation. |
| `--stride`             | 512          | Step of the sliding window. |
| `--dtype`              | float16      | Compute dtype: float16 / bfloat16 / float32. |
| `--text-file`          | None         | Evaluate on a user-provided `.txt` file. |
| `--hf-dataset`         | None         | `name:config:split`, e.g. `wikitext:wikitext-2-raw-v1:test`. |
| `--quant-group-size`   | 128          | GPTQ group size. |
| `--quant-bits`         | 4            | Number of bits (only 4 supported today). |
| `--skip-baseline`      | off          | Skip the FP run (quant-only). |
| `--skip-quant`         | off          | Skip the GPTQ run (baseline-only). |
