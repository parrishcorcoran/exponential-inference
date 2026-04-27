"""Test the shaped Qwen3-0.6B end-to-end:
  - Load shape-fine-tuned model + KV-Medusa head + K-decoder.
  - Run the KV-substitution token-acceptance test at offset 1 (oracle drafts).
  - Run the K-decoder readout test on PREDICTED K.
  - Compare to stock model's results.

Stock Qwen3-0.6B numbers for reference:
  - KV substitution top-1 acceptance: 0.96 at offset 1
  - K-decoder on predicted K: 0.21 top-1 (joint focused 1x1)
  - K-decoder on real K (ceiling): 0.56

The shape claim: cos_k 0.91 / cos_v 0.86 (vs stock 0.77 / 0.41) should push
both metrics up substantially.
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
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
N_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
SHAPED_PATH = CKPT_DIR / "qwen_06b_shaped.pt"
RESULTS_PATH = Path("results/pipeline_shaped_acceptance_test.json")


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads; self.head_dim = head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


class KDecoder(nn.Module):
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv, d_model, bias=False)
    def forward(self, K_flat, lm_head_weight):
        h = self.proj(K_flat)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


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
d_kv = n_kv_heads * head_dim
lm_head_weight = model.lm_head.weight.detach()

# Load shaped checkpoint
print("Loading shaped model + heads + decoder...")
shaped = torch.load(SHAPED_PATH, map_location=device)
model.load_state_dict(shaped["model"])
model.eval()

kv_head = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
kv_head.load_state_dict(shaped["kv_heads"][0])
kv_head.eval()

decoder = KDecoder(d_kv, d_model).to(device).to(torch.float32)
decoder.load_state_dict(shaped["decoder"])
decoder.eval()

print("Loading val tokens...")
val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

# ─── Patched layer-14 attention for KV substitution ────────────────────────
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

# ─── Test 1: KV substitution token acceptance (oracle drafts) ──────────────
print(f"\n{'='*60}\nTest 1: KV substitution token acceptance\n{'='*60}")

match_count = 0
total = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        baseline_top1 = out.logits.argmax(-1)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        with torch.no_grad():
            pk, pv = kv_head(h_t)
            sub_k = actual_k.clone()
            sub_v = actual_v.clone()
            sub_k[:, :, t + OFFSET, :] = pk[0, 0]
            sub_v[:, :, t + OFFSET, :] = pv[0, 0]
            sub_mask = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
            sub_mask[t + OFFSET] = True

            attn_layer._sub_k = sub_k; attn_layer._sub_v = sub_v; attn_layer._sub_mask = sub_mask
            out_sub = model(inp, use_cache=False)
            sub_top1 = out_sub.logits.argmax(-1)
            attn_layer._sub_k = None; attn_layer._sub_v = None; attn_layer._sub_mask = None

        pos = t + OFFSET
        if sub_top1[0, pos - 1] == baseline_top1[0, pos - 1]:
            match_count += 1
        total += 1

    print(f"  seq {seq_idx + 1}/{N_SEQS}", flush=True)

kv_sub_acc = match_count / total
print(f"\n  KV substitution acceptance (oracle drafts): {kv_sub_acc:.3f}")
print(f"  Stock model reference: 0.960")

# ─── Test 2: K-decoder readout on PREDICTED K ─────────────────────────────
print(f"\n{'='*60}\nTest 2: K-decoder readout on predicted K\n{'='*60}")

attn_layer.forward = orig_forward  # remove substitution patch (not needed for this test)

dec_match_top1 = dec_match_top5 = 0
total2 = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=False)
        h_final = out.hidden_states[-1].float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        with torch.no_grad():
            pk, _ = kv_head(h_t)
            K_flat_pred = pk.reshape(1, 1, d_kv)
            logits = decoder(K_flat_pred, lm_head_weight)
            top1 = logits.argmax(-1).item()
            top5 = set(logits.topk(5, dim=-1).indices[0, 0].tolist())

        true_tok = baseline_toks[t + OFFSET].item()
        if top1 == true_tok: dec_match_top1 += 1
        if true_tok in top5: dec_match_top5 += 1
        total2 += 1

dec_top1 = dec_match_top1 / total2
dec_top5 = dec_match_top5 / total2
print(f"\n  K-decoder predicted-K → token: top-1 {dec_top1:.3f}, top-5 {dec_top5:.3f}")
print(f"  Stock model reference (joint-1×1): top-1 0.21, top-5 0.46")

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SHAPED MODEL — END-TO-END TEST")
print(f"{'='*70}")
print(f"  KV-substitution acceptance: {kv_sub_acc:.3f}  (stock: 0.960)")
print(f"  Predicted-K → token top-1:  {dec_top1:.3f}  (stock: 0.21)")
print(f"  Predicted-K → token top-5:  {dec_top5:.3f}  (stock: 0.46)")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "shaped": str(SHAPED_PATH),
               "kv_sub_acceptance": round(kv_sub_acc, 4),
               "dec_predicted_top1": round(dec_top1, 4),
               "dec_predicted_top5": round(dec_top5, 4),
               "n_anchors": total}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
