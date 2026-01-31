
from transformers import AutoConfig
from dataclasses import dataclass

@dataclass
class Config:
    model_path: str
    max_num_batched_tokens: int = 16384  # max tokens in a batch
    max_num_seqs: int = 512  # max_seqs in a batch
    max_model_len: int = 4096  # max len of model
    gpu_memory_utilization: float = 0.9
    hf_config: AutoConfig | None = None
    kvcache_block_size: int = 256  # 必须是 256 的倍数，因为 flash-attn 的 block kvcache 是这样规定的
    num_kvcache_blocks: int = -1  # pre-allocate kvcache in wramup
    cuda_id: int = 3  # which gpu to use
    model_name: str = "Qwen3-0.6B-OPT"

    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)