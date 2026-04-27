"""Layer sweep: for each candidate input layer L, measure how well a small
probe predicts (K, V, Q) at layer-14, offset +1. Identify the optimal layer
to attach each of the K-head / V-head / Q-head.

Setup:
  - Sample layers: {3, 7, 11, 14, 18, 22, 26}.
  - For each layer L, train ONE multi-output probe: h_at_L -> (K, V, Q).
  - 500 steps per layer. MSE on each target. Measure cos_k, cos_v, cos_q.

Output: ranking of best layer per head type. We then plug that into the
final 1×1 K+V+Q architecture (heads at optimal spots, one decoder).
"""
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


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


def iter_batches(tokens, seq_len, device):
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < seq_len + 1: continue
        yield torch.tensor([window], dtype=torch.long, device=device)


class TripleHead(nn.Module):
    """Predicts K, V, Q simultaneously from a hidden state vector."""
    def __init__(self, d_model, n_kv_heads, n_attn_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads; self.n_attn_heads = n_attn_heads; self.head_dim = head_dim
        hidden = d_model // 2
        self.k = nn.Sequential(nn.Linear(d_model, hidden, bias=False), nn.SiLU(),
                               nn.Linear(hidden, n_kv_heads * head_dim, bias=False))
        self.v = nn.Sequential(nn.Linear(d_model, hidden, bias=False), nn.SiLU(),
                               nn.Linear(hidden, n_kv_heads * head_dim, bias=False))
        self.q = nn.Sequential(nn.Linear(d_model, hidden, bias=False), nn.SiLU(),
                               nn.Linear(hidden, n_attn_heads * head_dim, bias=False))
    def forward(self, h):
        b, s = h.shape[0], h.shape[1]
        return (self.k(h).view(b, s, self.n_kv_heads, self.head_dim),
                self.v(h).view(b, s, self.n_kv_heads, self.head_dim),
                self.q(h).view(b, s, self.n_attn_heads, self.head_dim))


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
SAMPLE_LAYERS = list(range(28))  # ALL layers
STEPS_PER_LAYER = 200
EVAL_EVERY = 100
LR = 5e-4
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_layer_sweep.json")


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
n_layers = model.config.num_hidden_layers

# ─── Capture Q at layer 14 ─────────────────────────────────────────────────
attn_layer = model.model.layers[TARGET_LAYER].self_attn
captured_Q = {}
orig_forward = attn_layer.forward

def capturing_forward(hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    self = attn_layer
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    qs = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    ks = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    cos, sin = position_embeddings
    qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
    captured_Q["q"] = qs.detach()
    return orig_forward(hidden_states, position_embeddings, attention_mask,
                        past_key_values, cache_position, **kwargs)

attn_layer.forward = capturing_forward

print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 1500)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 100, skip_tokens=SEQ_LEN * 1500)

print(f"Sweeping layers: {SAMPLE_LAYERS}")
print(f"  d_model={d_model}, n_kv={n_kv_heads}, n_attn={n_attn_heads}, head_dim={head_dim}")

results = []
for L in SAMPLE_LAYERS:
    print(f"\n{'='*60}\n  Layer {L} → predict (K, V, Q) at layer {TARGET_LAYER}, offset {OFFSET}\n{'='*60}")
    head = TripleHead(d_model, n_kv_heads, n_attn_heads, head_dim).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0
    for batch in iter_batches(train_tokens, SEQ_LEN, device):
        if step >= STEPS_PER_LAYER: break
        inp = batch[:, :SEQ_LEN]
        captured_Q.clear()
        with torch.no_grad():
            out = model(inp, use_cache=True, output_hidden_states=True)
            h_L = out.hidden_states[L].float()
            actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
            actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()
            actual_q = captured_Q["q"].float()
        h_in = h_L[:, :-OFFSET]
        target_k = actual_k[:, :, OFFSET:].permute(0, 2, 1, 3).float()
        target_v = actual_v[:, :, OFFSET:].permute(0, 2, 1, 3).float()
        target_q = actual_q[:, :, OFFSET:].permute(0, 2, 1, 3).float()
        ml = min(h_in.shape[1], target_k.shape[1])
        h_in, target_k, target_v, target_q = h_in[:, :ml], target_k[:, :ml], target_v[:, :ml], target_q[:, :ml]
        pk, pv, pq = head(h_in)
        loss = F.mse_loss(pk, target_k) + F.mse_loss(pv, target_v) + F.mse_loss(pq, target_q)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1

    # Val cosines
    head.eval()
    cs_k, cs_v, cs_q = [], [], []
    val_count = 0
    for vbatch in iter_batches(val_tokens, SEQ_LEN, device):
        if val_count >= 10: break
        vinp = vbatch[:, :SEQ_LEN]
        captured_Q.clear()
        with torch.no_grad():
            out = model(vinp, use_cache=True, output_hidden_states=True)
            h_L = out.hidden_states[L].float()
            ak = out.past_key_values.layers[TARGET_LAYER].keys.float()
            av = out.past_key_values.layers[TARGET_LAYER].values.float()
            aq = captured_Q["q"].float()
            h_in = h_L[:, :-OFFSET]
            tk = ak[:, :, OFFSET:].permute(0, 2, 1, 3).float()
            tv = av[:, :, OFFSET:].permute(0, 2, 1, 3).float()
            tq = aq[:, :, OFFSET:].permute(0, 2, 1, 3).float()
            ml = min(h_in.shape[1], tk.shape[1])
            pk, pv, pq = head(h_in[:, :ml])
            cs_k.append(F.cosine_similarity(pk.reshape(-1, head_dim), tk[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            cs_v.append(F.cosine_similarity(pv.reshape(-1, head_dim), tv[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            cs_q.append(F.cosine_similarity(pq.reshape(-1, head_dim), tq[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
        val_count += 1
    ck = sum(cs_k) / len(cs_k); cv = sum(cs_v) / len(cs_v); cq = sum(cs_q) / len(cs_q)
    print(f"  Layer {L} val: cos_k={ck:.3f}  cos_v={cv:.3f}  cos_q={cq:.3f}", flush=True)
    results.append({"layer": L, "cos_k": round(ck, 4), "cos_v": round(cv, 4), "cos_q": round(cq, 4)})
    del head, opt
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()

attn_layer.forward = orig_forward

print(f"\n{'='*60}\nLAYER SWEEP SUMMARY\n{'='*60}")
print(f"  {'layer':<8}{'cos_k':<10}{'cos_v':<10}{'cos_q':<10}")
for r in results:
    print(f"  L={r['layer']:<6}{r['cos_k']:<10.3f}{r['cos_v']:<10.3f}{r['cos_q']:<10.3f}")

best_k = max(results, key=lambda r: r["cos_k"])
best_v = max(results, key=lambda r: r["cos_v"])
best_q = max(results, key=lambda r: r["cos_q"])
print(f"\n  Best for K: layer {best_k['layer']} (cos_k={best_k['cos_k']:.3f})")
print(f"  Best for V: layer {best_v['layer']} (cos_v={best_v['cos_v']:.3f})")
print(f"  Best for Q: layer {best_q['layer']} (cos_q={best_q['cos_q']:.3f})")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER, "offset": OFFSET,
               "results": results, "best_k_layer": best_k["layer"],
               "best_v_layer": best_v["layer"], "best_q_layer": best_q["layer"]}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
