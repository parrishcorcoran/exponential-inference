"""
Holographic Multi-View Transformer.

Instead of L sequential layers (each a viewing angle on the manifold),
this model applies N parallel views with shared attention weights,
each rotated by the measured rotation schedule. Views are combined
via carry/flip decomposition.

Key architectural choices informed by findings:
- Finding 01: manifold is ~10D → d_manifold=16 (with headroom)
- Finding 02: rotation schedule is universal → use measured angles as fixed rotations
- Finding 10: bulk (MLP dim) is load-bearing → keep full d_int
- Finding 11: forward pass is RG flow → parallel views converge to same attractor
- Two-mode spectrum: carry + flip channels → coherent combination
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Standard RoPE for positional encoding within each view."""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, position_ids):
        # x: [B, n_heads, T, head_dim]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [B, T, dim/2]
        cos = freqs.cos().unsqueeze(1)  # [B, 1, T, dim/2]
        sin = freqs.sin().unsqueeze(1)
        cos = torch.cat([cos, cos], dim=-1)  # [B, 1, T, dim]
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin


def apply_rotary(x, cos, sin):
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class ManifoldView(nn.Module):
    """One viewing angle on the manifold.

    Applies a learned rotation to the input, then runs attention
    with shared Q/K/V weights. Each view sees the same data from
    a different angle.
    """
    def __init__(self, d_model, view_angle_init=0.0):
        super().__init__()
        # Learnable rotation parameters (Givens rotation in d_model space)
        # Initialize near the measured rotation angle for this view
        self.angle_scale = nn.Parameter(torch.tensor(view_angle_init))
        # Rotation is applied as: h_rotated = h * cos(angle) + h_perp * sin(angle)
        # where h_perp is a learned perpendicular direction
        self.perp_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.orthogonal_(self.perp_proj.weight)

    def rotate(self, h):
        """Apply view-specific rotation to hidden states."""
        angle = self.angle_scale
        h_perp = self.perp_proj(h)
        return h * angle.cos() + h_perp * angle.sin()


class SharedAttention(nn.Module):
    """Attention mechanism shared across all views.

    Q/K/V/O projections are shared — the difference between views
    comes from the rotation applied before attention.
    """
    def __init__(self, d_model, n_heads, head_dim, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads or n_heads
        self.kv_group = n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(head_dim)

    def forward(self, h, position_ids, attention_mask=None, past_kv=None):
        B, T, _ = h.shape

        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope(q, position_ids)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        # GQA expand
        if self.kv_group > 1:
            k = k.repeat_interleave(self.kv_group, dim=1)
            v = v.repeat_interleave(self.kv_group, dim=1)

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        # Causal mask
        T_q, T_kv = q.shape[2], k.shape[2]
        if T_q > 1:
            causal = torch.triu(torch.full((T_q, T_kv), float('-inf'),
                                           device=q.device), diagonal=T_kv - T_q + 1)
            scores = scores + causal

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_out), new_kv


class HolographicBlock(nn.Module):
    """The core holographic block: N parallel views + carry/flip combination + MLP.

    Replaces L sequential transformer layers with:
    1. N parallel viewing angles (shared attention weights)
    2. Carry/flip decomposition to combine views
    3. Single full-width MLP (bulk must be preserved)
    """
    def __init__(self, d_model, n_views, n_heads, head_dim, d_int, n_kv_heads=None):
        super().__init__()
        self.n_views = n_views
        self.d_model = d_model

        # Shared attention (one set of weights for all views)
        self.attention = SharedAttention(d_model, n_heads, head_dim, n_kv_heads)

        # Per-view rotations (different angles)
        # Initialize with spread of angles matching measured rotation schedule
        # Measured: layers rotate ~1.5 rad on average, spread across views
        angles = torch.linspace(0, math.pi * 0.8, n_views)
        self.views = nn.ModuleList([
            ManifoldView(d_model, angle.item()) for angle in angles
        ])

        # Per-view layer norms (different normalization per angle)
        self.view_norms = nn.ModuleList([
            nn.RMSNorm(d_model) for _ in range(n_views)
        ])

        # Carry/flip combination weights (learned)
        # Each view contributes to carry or flip channel
        self.view_weights = nn.Parameter(torch.ones(n_views) / n_views)
        self.carry_gate = nn.Linear(d_model, d_model, bias=False)
        nn.init.ones_(self.carry_gate.weight)  # start as identity

        # Post-attention norm
        self.post_attn_norm = nn.RMSNorm(d_model)

        # MLP (full d_int — bulk is load-bearing)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_int, bias=False)
        self.up_proj = nn.Linear(d_model, d_int, bias=False)
        self.down_proj = nn.Linear(d_int, d_model, bias=False)

    def forward(self, h, position_ids, past_kv=None):
        residual = h

        # Run N parallel views through shared attention
        view_outputs = []
        new_kvs = []
        for i, (view, norm) in enumerate(zip(self.views, self.view_norms)):
            # Rotate input to this view's angle
            h_rotated = view.rotate(norm(h))
            # Run through shared attention
            attn_out, new_kv = self.attention(h_rotated, position_ids, past_kv=past_kv)
            view_outputs.append(attn_out)
            new_kvs.append(new_kv)

        # Combine views via weighted sum (carry/flip decomposition)
        # Softmax over view weights for stable combination
        w = F.softmax(self.view_weights, dim=0)
        combined = sum(w[i] * view_outputs[i] for i in range(self.n_views))

        # Carry gate: controls how much of the original signal passes through
        carry = torch.sigmoid(self.carry_gate(residual))
        h = residual * carry + combined * (1 - carry)

        h = self.post_attn_norm(h)

        # MLP (single pass, full width)
        mlp_residual = h
        h_normed = self.mlp_norm(h)
        gate = F.silu(self.gate_proj(h_normed))
        up = self.up_proj(h_normed)
        h = mlp_residual + self.down_proj(gate * up)

        # Use the last view's KV cache (they share weights so any would work)
        return h, new_kvs[-1]


class HolographicTransformer(nn.Module):
    """Full holographic transformer.

    Architecture:
    - Embedding (shared with lm_head via weight tying)
    - M holographic blocks (each replaces ~L/M sequential layers)
    - Final norm + lm_head

    For a standard 28-layer transformer with M=2 blocks and N=8 views:
    - Each block's 8 parallel views replace ~14 sequential layers
    - Total "effective depth": 2 × 8 = 16 viewing angles
    - But only 2 sequential passes (vs 28)
    """
    def __init__(self, vocab_size, d_model, n_blocks, n_views, n_heads,
                 head_dim, d_int, n_kv_heads=None, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            HolographicBlock(d_model, n_views, n_heads, head_dim, d_int, n_kv_heads)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, position_ids=None, past_kvs=None, labels=None):
        B, T = input_ids.shape
        if position_ids is None:
            if past_kvs is not None and past_kvs[0] is not None:
                past_len = past_kvs[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + T,
                                           device=input_ids.device).unsqueeze(0).expand(B, -1)
            else:
                position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        h = self.embed(input_ids)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, kv = block(h, position_ids, past_kv=past_kv)
            new_kvs.append(kv)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size),
                                   shift_labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits, "past_kvs": new_kvs}

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Subtract tied weights (counted twice)
        tied = self.embed.weight.numel()
        return total - tied, trainable - tied

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        past_kvs = None
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            out = self.forward(generated if past_kvs is None else generated[:, -1:],
                              past_kvs=past_kvs)
            past_kvs = out["past_kvs"]
            logits = out["logits"][:, -1, :] / temperature
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 151645:  # Qwen EOS
                break

        return generated


def build_holographic_model(
    vocab_size=151936,  # Qwen3 tokenizer
    d_model=512,
    n_blocks=2,         # 2 sequential holographic blocks
    n_views=8,          # 8 parallel views per block (= 16 total viewing angles)
    n_heads=8,
    head_dim=64,
    d_int=2048,         # full MLP width (4x d_model, bulk preserved)
    n_kv_heads=4,       # GQA
):
    model = HolographicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_views=n_views,
        n_heads=n_heads,
        head_dim=head_dim,
        d_int=d_int,
        n_kv_heads=n_kv_heads,
    )
    n_params, n_trainable = model.count_params()
    print(f"Holographic Transformer: {n_params/1e6:.1f}M params ({n_trainable/1e6:.1f}M trainable)")
    print(f"  d_model={d_model}, n_blocks={n_blocks}, n_views={n_views}")
    print(f"  n_heads={n_heads}, head_dim={head_dim}, d_int={d_int}")
    print(f"  Effective viewing angles: {n_blocks * n_views}")
    print(f"  vs standard transformer: {n_blocks * n_views} views vs 28 sequential layers")
    return model
