from enum import Enum, auto
from itertools import count
from copy import copy

from src.sampling import SamplingParams

class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()) -> None:
        self.seq_id = next(Sequence.counter)
        self.status = SequenceState.WAITING
        self.token_ids = copy(token_ids)
        self.block_table = []
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.temprature = sampling_params.temprature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return len(self.token_ids)

    def is_finised(self) -> bool:
        return self.status == SequenceState.FINISHED
    
    def append(self, token_id: int): 
        self.token_ids.append(token_id)
    
    def block(self, i) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*Sequence.block_size: (i+1)*Sequence.block_size]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + Sequence.block_size - 1) // Sequence.block_size

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]
    
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_cached_blocks(self)->int:
        return self.num_cached_tokens // Sequence.block_size

    @property
    def num_last_block_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * Sequence.block_size
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def last_token(self):
        return self.token_ids[-1]
        