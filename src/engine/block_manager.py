from collections import deque
import xxhash
import numpy as np
from src.engine.Sequence import Sequence
from itertools import count
counter = count()

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).to_bytes())
        return h.intdigest()

    def _allocate_block(self, block_id) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            self.hash_to_block_id.pop(block.hash)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)

    def _deallocate_block(self, block_id) -> None:
        assert self.blocks[block_id].ref_count == 0, f"Block {block_id} still in use"
        self.free_block_ids.append(block_id)
        self.used_block_ids.remove(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self._deallocate_block(block_id)
            
        seq.block_table.clear()
        seq.num_cached_blocks = 0

    def allocate(self, seq: Sequence):
        assert not seq.block_table, "Sequence already has allocated blocks"
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == Sequence.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or token_ids != self.blocks[block_id].token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                self.hash_to_block_id[h] = block_id
                self.blocks[block_id].update(h, token_ids)

            else:
                seq.num_cached_tokens += Sequence.block_size
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                else:
                    self._allocate_block(block_id)
                    self.hash_to_block_id[h] = block_id
                    self.blocks[block_id].update(h, token_ids)

            seq.block_table.append(block_id)

    def append(self, seq: Sequence):
        block_table  = seq.block_table
        last_block = self.blocks[seq.block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2].hash] if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        
        else:
            assert last_block.hash == -1

    def can_allocate(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks
    
    def can_append(self, seq: Sequence):
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)