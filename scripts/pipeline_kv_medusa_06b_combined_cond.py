"""Combined acceptance test: token-Medusa drafts + *conditional* KV-Medusa cache.

Difference vs the unconditional combined test: the KV substitution at draft
position t+k uses head_k(h_t, embed(d_k)) — i.e., conditioned on the actual
drafted token. The hypothesis is that this cache-input consistency lets the
verify pass fairly evaluate even branches whose drafted token is "wrong"
(non-natural), so accept rates should rise above the unconditional baseline.

Reports:
  - raw token-Medusa accuracy
  - accept rate WITHOUT KV substitution
  - accept rate WITH UNCONDITIONAL KV substitution
  - accept rate WITH CONDITIONAL KV substitution
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
N_OFFSETS = 5
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_combined_cond.json")


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
        self.n_kv_heads = n_kv_heads; self.head_dim = head_dim

    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


class ConditionalKVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.k_pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.n_kv_heads = n_kv_heads; self.head_dim = head_dim

    def forward(self, h, token_embeds):
        x = torch.cat([h, token_embeds], dim=-1)
        k = self.k_pred(x).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(x).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
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
lm_head_weight = model.lm_head.weight.detach().float()
embed_layer = model.model.embed_tokens

print("Loading val tokens...")
val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

print(f"Loading heads...")
kv_heads = []
cond_kv_heads = []
tok_heads = []
for k in range(1, N_OFFSETS + 1):
    h1 = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h1.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h1.eval(); kv_heads.append(h1)

    h2 = ConditionalKVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h2.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_cond_head_{k}.pt", map_location=device))
    h2.eval(); cond_kv_heads.append(h2)

    h3 = MedusaHead(d_model, n_layers=1).to(device)
    h3.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h3.eval(); tok_heads.append(h3)

# ─── Patched layer-14 attention forward (KV substitution) ───────────────────
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
            key_states, value_states, self.layer_idx, cache_kwargs)

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
        dropout=0.0, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


attn_layer.forward = patched_forward
attn_layer._sub_k = attn_layer._sub_v = attn_layer._sub_mask = None

# ─── Run test ─────────────────────────────────────────────────────────────
no_sub = {k: 0 for k in range(1, N_OFFSETS + 1)}
uncond_sub = {k: 0 for k in range(1, N_OFFSETS + 1)}
cond_sub = {k: 0 for k in range(1, N_OFFSETS + 1)}
total = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue

        h_t = h_final[:, t:t+1]

        # Token-Medusa drafts
        with torch.no_grad():
            drafts = []
            for hd in tok_heads:
                logits = hd(h_t, lm_head_weight)
                drafts.append(logits.argmax(-1).item())
        drafts_t = torch.tensor(drafts, dtype=torch.long, device=device)

        # Unconditional KV-Medusa predictions
        sub_k_uncond = actual_k.clone()
        sub_v_uncond = actual_v.clone()
        with torch.no_grad():
            for k_idx, hd in enumerate(kv_heads):
                pk, pv = hd(h_t)
                sub_k_uncond[:, :, t + 1 + k_idx, :] = pk[0, 0]
                sub_v_uncond[:, :, t + 1 + k_idx, :] = pv[0, 0]

        # Conditional KV-Medusa predictions (conditioned on each drafted token)
        sub_k_cond = actual_k.clone()
        sub_v_cond = actual_v.clone()
        with torch.no_grad():
            for k_idx, hd in enumerate(cond_kv_heads):
                tok_id = drafts_t[k_idx:k_idx+1].view(1, 1)
                te = embed_layer(tok_id).float()
                pk, pv = hd(h_t, te)
                sub_k_cond[:, :, t + 1 + k_idx, :] = pk[0, 0]
                sub_v_cond[:, :, t + 1 + k_idx, :] = pv[0, 0]

        sub_mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
        sub_mask[t + 1: t + 1 + N_OFFSETS] = True

        verify_input = inp.clone()
        verify_input[0, t + 1: t + 1 + N_OFFSETS] = drafts_t

        # No KV substitution
        attn_layer._sub_k = None; attn_layer._sub_v = None; attn_layer._sub_mask = None
        with torch.no_grad():
            top1_a = model(verify_input, use_cache=False).logits.argmax(-1)

        # Unconditional KV substitution
        attn_layer._sub_k = sub_k_uncond; attn_layer._sub_v = sub_v_uncond; attn_layer._sub_mask = sub_mask
        with torch.no_grad():
            top1_b = model(verify_input, use_cache=False).logits.argmax(-1)

        # Conditional KV substitution
        attn_layer._sub_k = sub_k_cond; attn_layer._sub_v = sub_v_cond; attn_layer._sub_mask = sub_mask
        with torch.no_grad():
            top1_c = model(verify_input, use_cache=False).logits.argmax(-1)

        attn_layer._sub_k = None; attn_layer._sub_v = None; attn_layer._sub_mask = None

        for k in range(1, N_OFFSETS + 1):
            pos = t + k
            if pos >= SEQ_LEN: continue
            d_k = drafts[k - 1]
            if top1_a[0, pos - 1].item() == d_k: no_sub[k] += 1
            if top1_b[0, pos - 1].item() == d_k: uncond_sub[k] += 1
            if top1_c[0, pos - 1].item() == d_k: cond_sub[k] += 1
            total[k] += 1

    n_done += 1
    print(f"  seq {n_done}/{N_SEQS} done", flush=True)

attn_layer.forward = orig_forward

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("CONDITIONAL KV-MEDUSA + token-Medusa COMBINED TEST (Qwen3-0.6B)")
print(f"{'='*70}")
print(f"  N seqs: {N_SEQS}, anchors per seq: {len(ANCHORS)}")
print(f"\n  {'offset':<8}{'no KV':<10}{'uncond KV':<14}{'cond KV':<14}{'cond gain vs uncond':<22}")

results = []
for k in range(1, N_OFFSETS + 1):
    n = total[k]
    a = no_sub[k] / n if n else 0
    b = uncond_sub[k] / n if n else 0
    c = cond_sub[k] / n if n else 0
    print(f"  t+{k:<6}{a:<10.3f}{b:<14.3f}{c:<14.3f}{c-b:+.3f}")
    results.append({"offset": k, "n": n,
                    "accept_no_kv": round(a, 4),
                    "accept_uncond_kv": round(b, 4),
                    "accept_cond_kv": round(c, 4)})

def chain(rates):
    out = 1.0; prod = 1.0
    for r in rates:
        prod *= r; out += prod
    return out

print(f"\n  Chained tokens/step:")
print(f"    No KV substitution:           {chain([r['accept_no_kv'] for r in results]):.3f}")
print(f"    Unconditional KV-Medusa:      {chain([r['accept_uncond_kv'] for r in results]):.3f}")
print(f"    Conditional KV-Medusa:        {chain([r['accept_cond_kv'] for r in results]):.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_seqs": N_SEQS,
               "anchors_per_seq": len(ANCHORS), "results": results,
               "chained_no_kv": round(chain([r['accept_no_kv'] for r in results]), 4),
               "chained_uncond": round(chain([r['accept_uncond_kv'] for r in results]), 4),
               "chained_cond": round(chain([r['accept_cond_kv'] for r in results]), 4)}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
