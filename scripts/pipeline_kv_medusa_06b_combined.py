"""Combined acceptance test: 5 standard token-Medusa heads + 5 KV-Medusa heads.

Hypothesis: KV-Medusa cache substitution at draft positions should reduce the
cascading-error problem in standard Medusa, so token-acceptance with KV-Medusa
should be >= acceptance without KV-Medusa, especially at deeper offsets.

Per anchor t:
  1. Get baseline forward; save baseline tokens and h_t.
  2. Use 5 token-Medusa heads on h_t -> draft tokens d_1..d_5.
  3. Use 5 KV-Medusa heads on h_t -> predicted K, V at t+1..t+5 in layer 14.
  4. Build verify input: [prefix(0..t), d_1, d_2, d_3, d_4, d_5].
  5. Config A: forward without KV-substitution. Acceptance per offset:
     d_k accepted iff verify_logits[t+k-1].argmax() == d_k
  6. Config B: forward WITH KV-substitution at layer 14, positions t+1..t+5.
     Same acceptance check.

Also report:
  - Raw token-Medusa accuracy: P(d_k == baseline[t+k]) — baseline-matching
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
ANCHORS = [40, 80, 120, 160, 200]
TARGET_LAYER = 14
N_OFFSETS = 5  # we have 5 token-Medusa heads
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_combined.json")


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])

    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


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
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach().float()

print("Loading val tokens...")
val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

print(f"Loading {N_OFFSETS} KV-Medusa heads...")
kv_heads = []
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.eval(); kv_heads.append(h)

print(f"Loading {N_OFFSETS} token-Medusa heads...")
tok_heads = []
for k in range(1, N_OFFSETS + 1):
    h = MedusaHead(d_model, n_layers=1).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h.eval(); tok_heads.append(h)

# ─── Patched layer-14 attention forward (KV substitution) ──────────────────
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
        key_states = key_states.clone(); value_states = value_states.clone()
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
attn_layer._sub_k = attn_layer._sub_v = attn_layer._sub_mask = None

# ─── Run test ──────────────────────────────────────────────────────────────
raw_match = {k: 0 for k in range(1, N_OFFSETS + 1)}    # d_k == baseline[t+k]
no_sub_accept = {k: 0 for k in range(1, N_OFFSETS + 1)}  # logit no-sub agrees w/ d_k
with_sub_accept = {k: 0 for k in range(1, N_OFFSETS + 1)}
total = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        baseline_top1 = out.logits.argmax(-1)  # [1, seq] — natural greedy continuation
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue

        h_t = h_final[:, t:t+1]  # anchor

        # Token-Medusa drafts d_1..d_K from h_t
        with torch.no_grad():
            drafts = []
            for hd in tok_heads:
                logits = hd(h_t, lm_head_weight)  # [1, 1, vocab]
                drafts.append(logits.argmax(-1).item())
        drafts_t = torch.tensor(drafts, dtype=torch.long, device=device)

        # KV-Medusa predictions for layer-14 K, V at positions t+1..t+K
        sub_k = actual_k.clone()
        sub_v = actual_v.clone()
        with torch.no_grad():
            for k_idx, hd in enumerate(kv_heads):
                pk, pv = hd(h_t)
                sub_k[:, :, t + 1 + k_idx, :] = pk[0, 0]
                sub_v[:, :, t + 1 + k_idx, :] = pv[0, 0]

        sub_mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
        sub_mask[t + 1: t + 1 + N_OFFSETS] = True

        # Construct "verify input": replace tokens at t+1..t+N_OFFSETS with drafted tokens
        verify_input = inp.clone()
        verify_input[0, t + 1: t + 1 + N_OFFSETS] = drafts_t

        # Config A: no KV substitution
        attn_layer._sub_k = None
        attn_layer._sub_v = None
        attn_layer._sub_mask = None
        with torch.no_grad():
            out_a = model(verify_input, use_cache=False)
            top1_a = out_a.logits.argmax(-1)  # [1, seq]

        # Config B: with KV substitution at draft positions
        attn_layer._sub_k = sub_k
        attn_layer._sub_v = sub_v
        attn_layer._sub_mask = sub_mask
        with torch.no_grad():
            out_b = model(verify_input, use_cache=False)
            top1_b = out_b.logits.argmax(-1)
        attn_layer._sub_k = None; attn_layer._sub_v = None; attn_layer._sub_mask = None

        # Per-offset metrics
        for k in range(1, N_OFFSETS + 1):
            pos = t + k
            if pos >= SEQ_LEN: continue
            d_k = drafts[k - 1]
            base = baseline_top1[0, pos].item()
            # Acceptance check: model's logit at position pos-1 (which predicts token at pos)
            #                   argmax should equal d_k (drafted token at pos)
            no_sub_pred = top1_a[0, pos - 1].item()
            with_sub_pred = top1_b[0, pos - 1].item()

            if d_k == base: raw_match[k] += 1
            if no_sub_pred == d_k: no_sub_accept[k] += 1
            if with_sub_pred == d_k: with_sub_accept[k] += 1
            total[k] += 1

    n_done += 1
    print(f"  seq {n_done}/{N_SEQS} done", flush=True)

attn_layer.forward = orig_forward

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMBINED ACCEPTANCE TEST — token-Medusa + KV-Medusa, Qwen3-0.6B")
print(f"{'='*60}")
print(f"  N seqs: {N_SEQS}, anchors per seq: {len(ANCHORS)}")
print(f"\n  {'offset':<8}{'raw acc':<10}{'accept (no KV)':<18}{'accept (w/ KV)':<18}{'KV gain':<8}")

results = []
for k in range(1, N_OFFSETS + 1):
    n = total[k]
    raw = raw_match[k] / n if n else 0
    nos = no_sub_accept[k] / n if n else 0
    ws = with_sub_accept[k] / n if n else 0
    gain = ws - nos
    print(f"  t+{k:<6}{raw:<10.3f}{nos:<18.3f}{ws:<18.3f}{gain:+.3f}")
    results.append({
        "offset": k, "n": n,
        "raw_token_acc": round(raw, 4),
        "accept_no_kv": round(nos, 4),
        "accept_with_kv": round(ws, 4),
        "kv_gain": round(gain, 4),
    })

# Chained tokens-per-step under each scheme
def chain(rates):
    out = 1.0; prod = 1.0
    for a in rates:
        prod *= a; out += prod
    return out

raw_chain = chain([r["raw_token_acc"] for r in results])
nos_chain = chain([r["accept_no_kv"] for r in results])
ws_chain  = chain([r["accept_with_kv"] for r in results])

print(f"\n  Chained tokens/step (1 + a1 + a1*a2 + ...):")
print(f"    Raw token-Medusa accuracy:   {raw_chain:.3f}")
print(f"    Standard Medusa, no KV-sub:  {nos_chain:.3f}")
print(f"    Standard Medusa, with KV-sub: {ws_chain:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT, "n_seqs": N_SEQS, "anchors_per_seq": len(ANCHORS),
        "results": results,
        "chained_raw_acc": round(raw_chain, 4),
        "chained_accept_no_kv": round(nos_chain, 4),
        "chained_accept_with_kv": round(ws_chain, 4),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
