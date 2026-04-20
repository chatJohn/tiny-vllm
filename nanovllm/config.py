import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Quantization method applied after weights are loaded.
    # None -> no quantization (original fp weights).
    # "gptq" -> apply GPTQ int4 quantization to linear layers.
    quant_method: str | None = None
    quant_group_size: int = 128
    quant_bits: int = 4
    # KV-cache quantization: None keeps the cache in ``hf_config.torch_dtype``
    # (usually bf16/fp16); "int8" stores K/V as int8 + per-token-per-head fp
    # scales -- roughly halves the KV-cache memory footprint.
    kvcache_quant: str | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.quant_method in (None, "gptq"), (
            f"quant_method must be None or 'gptq', got {self.quant_method!r}"
        )
        assert self.kvcache_quant in (None, "int8"), (
            f"kvcache_quant must be None or 'int8', got {self.kvcache_quant!r}"
        )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
