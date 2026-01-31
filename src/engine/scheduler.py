from collections import deque
from transformers import AutoTokenizer

from src.config import Config
from src.sampling import SamplingParams
from src.engine.Sequence import Sequence, SequenceState
from src.engine.block_manager import BlockManager


def Scheduler:
    def __init__(self, config: Config) -> None:
        self.max_num_seqs = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
        self.eos = self.tokenizer.eos_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting = deque()
        self.running = deque()

    def add_request(self, prompt_token_ids, sampling_params):
        assert len(prompt_token_ids) < self.max_model_len
        seq = Sequence(prompt_token_ids, sampling_params)
        self.waiting.append(seq)

    def is_finised(self):
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            seq_len = len(seq)
            if num_batched_tokens + seq_len > self.max_num_batched_tokens:
                break
            numseqs += 1
            num_batched_tokens += seq_len - seq.num_cached_tokens
            self.block_manager.allocate(seq)
            scheduled_seqs.append(seq)
            seq.status = SequenceState.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # prefill
        
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                num_batched_tokens += 1
                self.block_manager.append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceState.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append(token_id)
            if ((not seq.ignore_eos and token_id == self.eos) or
                (seq.num_completion_tokens == seq.max_tokens) or
                (self.max_model_len == seq.num_tokens)):
                seq.status == SequenceState.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                