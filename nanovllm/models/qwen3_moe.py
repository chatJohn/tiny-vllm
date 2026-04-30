import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope

class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3MoeConfig, tp_group=None, ep_group=None) -> None:
        super().__init__()
        from nanovllm.layers.linear import set_tp_group
        set_tp_group(tp_group)

        self.model = Qwen3MoeModel(config, ep_group=ep_group)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits



class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        ep_group=None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx, ep_group=ep_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int = -1,
        ep_group=None,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config, ep_group=ep_group)
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

        
class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    MoE block with optional Expert Parallel support.
    When ep_group is None or ep_size = 1, this is a normal MoE block.
    When ep_size > 1 each rank holds only (num_experts // ep_size) experts.
    A two-phase all-to-all is used:
        1. Dispatch - send each token to the top k experts which are picked by the router
        2. Combine - send the computed results back to the original rank
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        ep_group = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=self.ep_group) if self.ep_group is not None else 1
        self.ep_rank = dist.get_rank(group=self.ep_group) if self.ep_group is not None else 0
        self.experts_per_rank = self.num_experts // self.ep_size

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                )
                for _ in range(self.experts_per_rank)
            ]
        )
    
    def _is_local_expert(self, global_expert_idx: int) -> bool:
        start = self.ep_rank * self.experts_per_rank
        end = start + self.experts_per_rank
        return start <= global_expert_idx < end
    
    def _global_to_local_expert(self, global_expert_idx: int) -> int:
        return global_expert_idx - self.ep_rank * self.experts_per_rank
    
    def _forward_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim = -1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(
            hidden_states,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        return final_hidden_states
    
    def _forward_ep(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim = 1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim = -1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # ---- Phase 1: Dispatch (all-to-all) ----
        # For each of the top_k slots, figure out which EP rank owns that expert.
        # expert_rank[i, k] = selected_experts[i, k] // experts_per_rank
        expert_rank = selected_experts // self.experts_per_rank

        # Build send buffers: one list entry per destination EP rank.
        # send_tokens[r]  : hidden states of tokens whose k-th expert lives on rank r
        # send_weights[r] : corresponding routing weights
        # send_meta[r]    : (original_token_idx, top_k_slot, local_expert_idx)
        send_tokens = [[] for _ in range(self.ep_size)]
        send_weights = [[] for _ in range(self.ep_size)]
        send_meta = [[] for _ in range(self.ep_size)]

        for k in range(self.top_k):
            for i in range(seq_len):
                r = expert_rank[i, k].item()
                local_expert = selected_experts[i, k] % self.experts_per_rank
                send_tokens[r].append(hidden_states[i])
                send_weights[r].append(routing_weights[i, k])
                send_meta[r].append((i, local_expert))

        send_counts = torch.tensor(
            [len(send_tokens[r]) for r in range(self.ep_size)],
            dtype=torch.int32,
            device=hidden_states.device,
        )
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        flat_send_tokens = torch.stack(
            [t for r in range(self.ep_size) for t in send_tokens[r]]
        ) if sum(len(send_tokens[r]) for r in range(self.ep_size)) > 0 else \
            torch.empty((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        flat_send_weights = torch.stack(
            [w for r in range(self.ep_size) for w in send_weights[r]]
        ) if sum(len(send_weights[r]) for r in range(self.ep_size)) > 0 else \
            torch.empty((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        flat_send_meta = torch.tensor(
            [m for r in range(self.ep_size) for m in send_meta[r]],
            dtype=torch.int32,
            device=hidden_states.device,
        ) if sum(len(send_meta[r]) for r in range(self.ep_size)) > 0 else \
            torch.empty((0, 2), dtype=torch.int32, device=hidden_states.device)
        
        total_recv = recv_counts.sum().item()

        # all-to-all for token hidden states
        flat_recv_tokens = torch.empty(
            total_recv, 
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_to_all_single(
            flat_recv_tokens, flat_send_tokens, 
            output_split_sizes=recv_counts.tolist(), 
            input_split_sizes=send_counts.tolist(), 
            group=self.ep_group
        )
        # all-to-all for token routing weights
        flat_recv_weights = torch.empty(
            total_recv,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_to_all_single(
            flat_recv_weights, flat_send_weights, 
            output_split_sizes=recv_counts.tolist(), 
            input_split_sizes=send_counts.tolist(), 
            group=self.ep_group
        )
        # all-to-all for token meta data
        flat_recv_meta = torch.empty(
            total_recv, 2, dtype=torch.int32, device=hidden_states.device
        )
        dist.all_to_all_single(
            flat_recv_meta, flat_send_meta, 
            output_split_sizes=recv_counts.tolist(), 
            input_split_sizes=send_counts.tolist(), 
            group=self.ep_group
        )
        # Local expert foward
        local_output = torch.zeros_like(flat_recv_tokens)
        if total_recv > 0:
            local_expert_ids = flat_recv_meta[:, 1]
            for local_expert_id in range(self.experts_per_rank):
                mask = local_expert_ids == local_expert_id
                if not mask.any():
                    continue
                inp = flat_recv_tokens[mask]
                out = self.experts[local_expert_id](inp)
                local_output[mask] = out
        
        local_output = local_output * flat_recv_weights.unsqueeze(-1)
        # ---- Phase 2: Combine (all-to-all) ----
        flat_combined = torch.zeros_like(flat_send_tokens)
        dist.all_to_all_single(
            flat_combined, local_output, 
            output_split_sizes=send_counts.tolist(), 
            input_split_sizes=recv_counts.tolist(), 
            group=self.ep_group
        )
        final_hidden_states = torch.zeros_like(hidden_states)
        offset = 0
        for r in range(self.ep_size):
            cnt = send_counts[r].item()
            for j in range(cnt):
                origin_idx = send_meta[r][j][0]
                final_hidden_states[origin_idx] = flat_combined[offset + j]
            offset += cnt
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor):
        if self.ep_size == 1:
            return self._forward_local(hidden_states)
        else:
            return self._forward_ep(hidden_states)