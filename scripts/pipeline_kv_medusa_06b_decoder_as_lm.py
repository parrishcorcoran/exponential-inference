"""Decoder-as-LM test: use K+V+Q decoder as autoregressive next-token producer.

Each generation step:
  1. Run main model on current prompt -> hidden_states[14], hidden_states[15]; capture Q at L14.
  2. K-head, V-head on h_at_L14[-1]; Q-head on h_at_L15[-1].
  3. Decoder reads (K_pred, V_pred, Q_pred) -> token logits.
  4. Argmax or sample -> next token.
  5. Append; repeat.

Measure:
  - Sample text (qualitative)
  - Self-PPL of decoder text under the original model
  - Side-by-side vs baseline greedy decode
"""
import json
import math
from pathlib import Path

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


CHECKPOINT = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
INPUT_LAYER_KV = 14
INPUT_LAYER_Q = 15
N_GENERATE = 80
PROMPTS = [
    "The future of artificial intelligence depends on",
    "In the early morning, the city was quiet, and",
    "Once upon a time, there was a small village where",
]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_decoder_as_lm.json")


class HeadOut(nn.Module):
    def __init__(self, d_model, n_out_heads, head_dim):
        super().__init__()
        self.n_out_heads = n_out_heads; self.head_dim = head_dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_out_heads * head_dim, bias=False),
        )
    def forward(self, h):
        return self.proj(h).view(h.shape[0], h.shape[1], self.n_out_heads, self.head_dim)


class WhitenedKVQDecoder(nn.Module):
    def __init__(self, d_kv, d_q, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv + d_kv + d_q, d_model, bias=False)
        self.register_buffer("W_K", torch.eye(d_kv))
        self.register_buffer("W_V", torch.eye(d_kv))
        self.register_buffer("W_Q", torch.eye(d_q))
        self.register_buffer("mu_K", torch.zeros(d_kv))
        self.register_buffer("mu_V", torch.zeros(d_kv))
        self.register_buffer("mu_Q", torch.zeros(d_q))
    def forward(self, K_flat, V_flat, Q_flat, lm_head_weight):
        K_w = (K_flat - self.mu_K) @ self.W_K
        V_w = (V_flat - self.mu_V) @ self.W_V
        Q_w = (Q_flat - self.mu_Q) @ self.W_Q
        x = torch.cat([K_w, V_w, Q_w], dim=-1)
        h = self.proj(x)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


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
d_kv = n_kv_heads * head_dim
d_q = n_attn_heads * head_dim
lm_head_weight = model.lm_head.weight.detach()

# ─── Hook for Q ────────────────────────────────────────────────────────────
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

# ─── Load trained heads + whitened decoder ─────────────────────────────────
print("Loading heads + decoder...")
k_head = HeadOut(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
v_head = HeadOut(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
q_head = HeadOut(d_model, n_attn_heads, head_dim).to(device).to(torch.float32)
k_head.load_state_dict(torch.load(CKPT_DIR / "k_head_kvq_whitened.pt", map_location=device))
v_head.load_state_dict(torch.load(CKPT_DIR / "v_head_kvq_whitened.pt", map_location=device))
q_head.load_state_dict(torch.load(CKPT_DIR / "q_head_kvq_whitened.pt", map_location=device))
k_head.eval(); v_head.eval(); q_head.eval()

decoder = WhitenedKVQDecoder(d_kv, d_q, d_model).to(device).to(torch.float32)
dec_state = torch.load(CKPT_DIR / "decoder_kvq_whitened.pt", map_location=device)
decoder.proj.weight.data.copy_(dec_state["proj_weight"])
decoder.W_K.copy_(dec_state["W_K"]); decoder.W_V.copy_(dec_state["W_V"]); decoder.W_Q.copy_(dec_state["W_Q"])
decoder.mu_K.copy_(dec_state["mu_K"]); decoder.mu_V.copy_(dec_state["mu_V"]); decoder.mu_Q.copy_(dec_state["mu_Q"])
decoder.eval()


@torch.no_grad()
def generate_baseline(prompt, n_tokens):
    """Standard greedy: model.lm_head(h_final).argmax()."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out_ids = ids.clone()
    for _ in range(n_tokens):
        out = model(out_ids, use_cache=False)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        out_ids = torch.cat([out_ids, next_tok], dim=1)
    return out_ids


@torch.no_grad()
def generate_decoder_lm(prompt, n_tokens):
    """Decoder-as-LM: K+V+Q decoder produces each next token."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out_ids = ids.clone()
    for _ in range(n_tokens):
        captured_Q.clear()
        out = model(out_ids, output_hidden_states=True, use_cache=False)
        h_l14 = out.hidden_states[INPUT_LAYER_KV][:, -1:].float()
        h_l15 = out.hidden_states[INPUT_LAYER_Q][:, -1:].float()
        pk = k_head(h_l14)
        pv = v_head(h_l14)
        pq = q_head(h_l15)
        pk_flat = pk.reshape(1, 1, d_kv)
        pv_flat = pv.reshape(1, 1, d_kv)
        pq_flat = pq.reshape(1, 1, d_q)
        logits = decoder(pk_flat, pv_flat, pq_flat, lm_head_weight)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        out_ids = torch.cat([out_ids, next_tok], dim=1)
    return out_ids


@torch.no_grad()
def self_perplexity(ids):
    """PPL of the sequence under the original model — how surprised is the model?"""
    out = model(ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item(), math.exp(nll.mean().item())


# ─── Run ──────────────────────────────────────────────────────────────────
results = []
for prompt in PROMPTS:
    print(f"\n{'='*70}\nPROMPT: {prompt!r}\n{'='*70}")

    base_ids = generate_baseline(prompt, N_GENERATE)
    dec_ids = generate_decoder_lm(prompt, N_GENERATE)

    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    dec_text = tokenizer.decode(dec_ids[0], skip_special_tokens=True)

    base_nll, base_ppl = self_perplexity(base_ids)
    dec_nll, dec_ppl = self_perplexity(dec_ids)

    print(f"\n--- BASELINE (greedy LM head) ---")
    print(base_text)
    print(f"\nself-PPL: {base_ppl:.2f}, mean NLL: {base_nll:.3f}")

    print(f"\n--- DECODER-AS-LM (K+V+Q whitened) ---")
    print(dec_text)
    print(f"\nself-PPL: {dec_ppl:.2f}, mean NLL: {dec_nll:.3f}")

    results.append({"prompt": prompt,
                    "baseline_text": base_text, "baseline_ppl": base_ppl, "baseline_nll": base_nll,
                    "decoder_text": dec_text, "decoder_ppl": dec_ppl, "decoder_nll": dec_nll})

attn_layer.forward = orig_forward

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_generate": N_GENERATE, "results": results}, f, indent=2)
print(f"\n\nSaved {RESULTS_PATH}")
