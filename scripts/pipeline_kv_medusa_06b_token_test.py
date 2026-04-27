"""KV-Medusa 0.6B — token-level acceptance test.

For each anchor position t in a held-out sequence, use h_t (final hidden) to
predict layer-14 K,V at positions t+1..t+10 via the 10 trained heads. Substitute
these into layer-14's K,V at the post-RoPE / post-norm point (where the heads'
training target lives). Run forward; compare top-1 tokens at t+1..t+10 to the
baseline run.

Per-offset acceptance is measured independently across many anchor positions.
The chained metric (1 + a1 + a1*a2 + ...) is the standard speculative-decoding
upper bound assuming offset matches are independent.
"""
import torch
import torch.nn as nn
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
N_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]  # 5 anchors per seq
TARGET_LAYER = 14
N_OFFSETS = 10
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_token_test.json")


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )

    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print(f"Loading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)

print(f"Loading val tokens (skip {SEQ_LEN * 4000})...", flush=True)
val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

print(f"Loading {N_OFFSETS} heads...", flush=True)
heads = {}
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.eval()
    heads[k] = h

# ─────────────────────────────────────────────────────────────────────
# Patched layer-14 attention forward (substitutes K, V at masked positions)
# ─────────────────────────────────────────────────────────────────────
attn_layer = model.model.layers[TARGET_LAYER].self_attn
orig_forward = attn_layer.forward


def patched_forward(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kwargs):
    self = attn_layer
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    sub_k = getattr(self, "_sub_k", None)
    sub_v = getattr(self, "_sub_v", None)
    sub_mask = getattr(self, "_sub_mask", None)
    if sub_k is not None and sub_mask is not None and sub_mask.any():
        key_states = key_states.clone()
        value_states = value_states.clone()
        key_states[:, :, sub_mask, :] = sub_k[:, :, sub_mask, :].to(key_states.dtype)
        value_states[:, :, sub_mask, :] = sub_v[:, :, sub_mask, :].to(value_states.dtype)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


attn_layer.forward = patched_forward
attn_layer._sub_k = None
attn_layer._sub_v = None
attn_layer._sub_mask = None

# ─────────────────────────────────────────────────────────────────────
# Run test
# ─────────────────────────────────────────────────────────────────────
per_offset_match = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_total = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1:
        break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)  # [1, SEQ_LEN]

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        baseline_top1 = out.logits.argmax(-1)  # [1, seq]
        h_final = out.hidden_states[-1].float()  # [1, seq, d]
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN:
            continue

        # Build sub_k, sub_v: start from actual; replace positions [t+1..t+N_OFFSETS]
        sub_k = actual_k.clone()
        sub_v = actual_v.clone()
        h_t = h_final[:, t:t+1]  # [1, 1, d] — anchor hidden state

        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                pk, pv = heads[k](h_t)  # [1, 1, n_kv, head_dim]
                sub_k[:, :, t + k, :] = pk[0, 0]
                sub_v[:, :, t + k, :] = pv[0, 0]

        sub_mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
        sub_mask[t + 1: t + 1 + N_OFFSETS] = True

        attn_layer._sub_k = sub_k
        attn_layer._sub_v = sub_v
        attn_layer._sub_mask = sub_mask

        with torch.no_grad():
            out_sub = model(inp, use_cache=False)
            sub_top1 = out_sub.logits.argmax(-1)

        # At position p, logit predicts token at p+1. We compare top-1 at positions t+1..t+N_OFFSETS
        # since substituted KV at positions t+1..t+N_OFFSETS affect attention at positions >= t+1.
        for k in range(1, N_OFFSETS + 1):
            pos = t + k
            if pos >= SEQ_LEN:
                continue
            base_tok = baseline_top1[0, pos].item()
            sub_tok = sub_top1[0, pos].item()
            if base_tok == sub_tok:
                per_offset_match[k] += 1
            per_offset_total[k] += 1

        attn_layer._sub_k = None
        attn_layer._sub_v = None
        attn_layer._sub_mask = None

    n_done += 1
    print(f"  seq {n_done}/{N_SEQS} done", flush=True)

# Restore
attn_layer.forward = orig_forward

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("KV-MEDUSA TOKEN-ACCEPTANCE TEST (Qwen3-0.6B, layer 14)")
print(f"{'=' * 60}")
print(f"  N seqs: {N_SEQS}, anchors per seq: {len(ANCHORS)}, seq_len: {SEQ_LEN}")

results = []
for k in range(1, N_OFFSETS + 1):
    rate = per_offset_match[k] / per_offset_total[k] if per_offset_total[k] > 0 else 0.0
    print(f"  t+{k:>2}: top-1 match {per_offset_match[k]:>3}/{per_offset_total[k]:>3} = {rate:.3f}")
    results.append({
        "offset": k,
        "match": per_offset_match[k],
        "total": per_offset_total[k],
        "accept_rate": round(rate, 4),
    })

# Independent and chained estimates
expected_indep = sum(r["accept_rate"] for r in results)
chained = 1.0
prod = 1.0
for r in results:
    prod *= r["accept_rate"]
    chained += prod

print(f"\n  Sum of independent rates: {expected_indep:.3f}")
print(f"  Chained tokens/step (1 + a1 + a1*a2 + ...): {chained:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_seqs": N_SEQS,
        "anchors_per_seq": len(ANCHORS),
        "seq_len": SEQ_LEN,
        "target_layer": TARGET_LAYER,
        "results": results,
        "sum_independent_rates": round(expected_indep, 4),
        "chained_tokens_per_step": round(chained, 4),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
