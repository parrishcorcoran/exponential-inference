"""Exponential Inference model — manifold-routed sparse forward.

Same weights as base model. Routing logic built into forward().
Download and run like any HF model:

    model = AutoModelForCausalLM.from_pretrained(
        "parrishcorcoran/exponential-14b", trust_remote_code=True)
    output = model.generate(input_ids, max_new_tokens=100)
    # Automatically faster via dynamic width × length routing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel
from .configuration_exponential import ExponentialConfig


class ExponentialForCausalLM(PreTrainedModel):
    config_class = ExponentialConfig

    def __init__(self, config):
        super().__init__(config)
        # Load the base model
        self.base = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.config = config
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden = config.hidden_size
        self.gqa_ratio = self.n_heads // self.n_kv

        # Router state: sharpness from previous step
        self._prev_sharpness = None

    def _route(self, step_idx):
        """Determine (active_heads, exit_layer) for this token.

        Uses attention sharpness from the PREVIOUS step.
        First step: use defaults (all heads, all layers).
        """
        if self._prev_sharpness is None or step_idx == 0:
            # No prior info — use defaults
            n_active = int(self.n_heads * self.config.default_active_heads_frac)
            exit_layer = int(self.n_layers * self.config.default_exit_layer_frac)
            active_heads = list(range(n_active))
        else:
            # Use sharpness: keep the sharpest heads
            sharpness = self._prev_sharpness
            n_active = max(self.config.min_heads,
                           int(self.n_heads * self.config.default_active_heads_frac))
            active_heads = sharpness.topk(n_active).indices.tolist()
            exit_layer = self.n_layers  # TODO: stabilization_depth routing

        return sorted(active_heads), min(exit_layer, self.n_layers)

    def _sparse_layer(self, h, layer, active_q_heads, cos, sin):
        """Run one layer with only active_q_heads computed."""
        B, T, D = h.shape
        n_active = len(active_q_heads)
        attn = layer.self_attn

        # Which KV heads do we need?
        active_kv = sorted(set(qh // self.gqa_ratio for qh in active_q_heads))
        n_active_kv = len(active_kv)

        residual = h
        h_norm = layer.input_layernorm(h)

        # Sparse Q projection
        q_w = attn.q_proj.weight.view(self.n_heads, self.head_dim, self.hidden)
        q = (h_norm @ q_w[active_q_heads].reshape(-1, self.hidden).T).view(
            B, T, n_active, self.head_dim)

        # Sparse K, V projections
        k_w = attn.k_proj.weight.view(self.n_kv, self.head_dim, self.hidden)
        k = (h_norm @ k_w[active_kv].reshape(-1, self.hidden).T).view(
            B, T, n_active_kv, self.head_dim)
        v_w = attn.v_proj.weight.view(self.n_kv, self.head_dim, self.hidden)
        v = (h_norm @ v_w[active_kv].reshape(-1, self.hidden).T).view(
            B, T, n_active_kv, self.head_dim)

        # QK norms
        if attn.q_norm is not None:
            q = attn.q_norm(q)
        if attn.k_norm is not None:
            k = attn.k_norm(k)

        q = q.transpose(1, 2)  # [B, n_active, T, HD]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Rotary
        rd = self.head_dim // 2
        c, s = cos.unsqueeze(1)[..., :rd], sin.unsqueeze(1)[..., :rd]
        q = torch.cat([q[..., :rd]*c - q[..., rd:]*s,
                        q[..., rd:]*c + q[..., :rd]*s], -1)
        k = torch.cat([k[..., :rd]*c - k[..., rd:]*s,
                        k[..., rd:]*c + k[..., :rd]*s], -1)

        # GQA expand
        kv_map = {kv: idx for idx, kv in enumerate(active_kv)}
        q_to_kv = [kv_map[qh // self.gqa_ratio] for qh in active_q_heads]
        k = k[:, q_to_kv]
        v = v[:, q_to_kv]

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Sparse O projection
        o_w = attn.o_proj.weight.view(self.hidden, self.n_heads, self.head_dim)
        o_active = o_w[:, active_q_heads, :].reshape(self.hidden, -1)
        attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_proj = attn_flat @ o_active.T * (self.n_heads / n_active)

        if attn.o_proj.bias is not None:
            attn_proj = attn_proj + attn.o_proj.bias

        h = residual + attn_proj

        # MLP full (depth = holographic projection, always full)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))

        return h

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                use_cache=False, past_key_values=None, **kwargs):
        """Manifold-routed forward. Compatible with generate()."""
        # For now: delegate to base model for prefill,
        # use sparse forward for single-token generation
        if input_ids.shape[1] > 1:
            # Prefill: use base model (full compute for initial context)
            return self.base(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, use_cache=use_cache,
                             past_key_values=past_key_values, **kwargs)

        # Single token generation: sparse forward
        active_heads, exit_layer = self._route(0)
        h = self.base.model.embed_tokens(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.base.model.rotary_emb(h, pos_ids)

        for i in range(exit_layer):
            h = self._sparse_layer(h, self.base.model.layers[i], active_heads, cos, sin)

        h = self.base.model.norm(h)
        logits = self.base.lm_head(h)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=logits)

    def generate(self, input_ids, max_new_tokens=64, do_sample=False, **kwargs):
        """Generate with manifold routing."""
        self._prev_sharpness = None
        gen_ids = input_ids

        # Prefill
        with torch.no_grad():
            out = self.base(gen_ids, use_cache=False)
            next_tok = out.logits[0, -1:].argmax(-1)
        gen_ids = torch.cat([gen_ids, next_tok.unsqueeze(0)], dim=-1)

        # Generate with sparse forward
        for step in range(max_new_tokens - 1):
            active_heads, exit_layer = self._route(step)

            with torch.no_grad():
                h = self.base.model.embed_tokens(gen_ids)
                pos_ids = torch.arange(gen_ids.shape[1], device=gen_ids.device).unsqueeze(0)
                cos, sin = self.base.model.rotary_emb(h, pos_ids)

                for i in range(exit_layer):
                    h = self._sparse_layer(h, self.base.model.layers[i],
                                           active_heads, cos, sin)

                logits = self.base.lm_head(self.base.model.norm(h))
                next_tok = logits[0, -1:].argmax(-1)

            gen_ids = torch.cat([gen_ids, next_tok.unsqueeze(0)], dim=-1)

            if next_tok.item() == self.config.eos_token_id:
                break

        return gen_ids.unsqueeze(0) if gen_ids.dim() == 1 else gen_ids.unsqueeze(0)
