"""K + V + Q decoder with whitening + warm-starts. Optimal-layer placement.

Adds to kvq_optimal:
  1. Whitening matrices W_K, W_V, W_Q computed from real K, V, Q at layer 14
     on calibration data. Σ⁻¹/² applied before decoder. Pushes predictions
     toward isotropic / HRR-unitary geometry.
  2. Warm-start ALL components — K/V heads from joint-1×1 checkpoint, Q-head
     from the partial training we just did, decoder from same.
  3. Epoch loop so the 2000-step budget actually runs.

Architecture:
  K-head reads h_at_L14 → predicts K
  V-head reads h_at_L14 → predicts V
  Q-head reads h_at_L15 → predicts Q
  Decoder reads [W_K @ K_pred ; W_V @ V_pred ; W_Q @ Q_pred] → token
"""
import json
import random
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


def epoch_iter(tokens, seq_len, device):
    """Cycles through dataset indefinitely."""
    while True:
        yield from iter_batches(tokens, seq_len, device)


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
        # Whitening matrices stored as buffers — not trained
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


def compute_whitening(samples, eps=1e-3):
    """ZCA whitening: Σ⁻¹/² via eigendecomposition with stabilization.
    eigh runs on CPU (not implemented on MPS)."""
    samples = samples.reshape(-1, samples.shape[-1]).float().cpu()  # [N, d]
    mu = samples.mean(0)
    centered = samples - mu
    Sigma = centered.T @ centered / centered.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = torch.clamp(eigvals, min=eps)
    inv_sqrt = eigvecs @ torch.diag(eigvals ** -0.5) @ eigvecs.T
    return inv_sqrt, mu


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
INPUT_LAYER_KV = 14
INPUT_LAYER_Q = 15
OFFSET = 1
TRAIN_STEPS = 2000
EVAL_EVERY = 100
LR = 1e-4  # lower since warm-started
N_CALIB = 30  # batches for whitening calibration
N_EVAL_SEQS = 20
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_kvq_whitened.json")


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
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

print(f"  d_model={d_model}, d_kv={d_kv}, d_q={d_q}")

# ─── Hook to capture Q ─────────────────────────────────────────────────────
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
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

# ─── Calibration: collect K, V, Q samples; compute whitening ───────────────
print(f"\nComputing whitening matrices on {N_CALIB} calibration batches...")
K_samples, V_samples, Q_samples = [], [], []
calib_count = 0
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if calib_count >= N_CALIB: break
    inp = batch[:, :SEQ_LEN]
    captured_Q.clear()
    with torch.no_grad():
        out = model(inp, use_cache=True, output_hidden_states=False)
        K = out.past_key_values.layers[TARGET_LAYER].keys.float()
        V = out.past_key_values.layers[TARGET_LAYER].values.float()
        Q = captured_Q["q"].float()
        K_flat = K.permute(0, 2, 1, 3).reshape(-1, d_kv)
        V_flat = V.permute(0, 2, 1, 3).reshape(-1, d_kv)
        Q_flat = Q.permute(0, 2, 1, 3).reshape(-1, d_q)
        K_samples.append(K_flat)
        V_samples.append(V_flat)
        Q_samples.append(Q_flat)
    calib_count += 1

K_all = torch.cat(K_samples, 0)
V_all = torch.cat(V_samples, 0)
Q_all = torch.cat(Q_samples, 0)

W_K, mu_K = compute_whitening(K_all)
W_V, mu_V = compute_whitening(V_all)
W_Q, mu_Q = compute_whitening(Q_all)

print(f"  K samples: {K_all.shape[0]}, condition number: {torch.linalg.cond(W_K).item():.1e}")
print(f"  V samples: {V_all.shape[0]}, condition number: {torch.linalg.cond(W_V).item():.1e}")
print(f"  Q samples: {Q_all.shape[0]}, condition number: {torch.linalg.cond(W_Q).item():.1e}")
del K_samples, V_samples, Q_samples, K_all, V_all, Q_all

# ─── Heads + decoder, all warm-started ─────────────────────────────────────
k_head = HeadOut(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
v_head = HeadOut(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
q_head = HeadOut(d_model, n_attn_heads, head_dim).to(device).to(torch.float32)

# K, V from joint-1×1 (focused training)
joint_state = torch.load(CKPT_DIR / "kv_medusa_head_joint_one_1.pt", map_location=device)
k_state = {n.replace("k_pred.", "proj."): vv for n, vv in joint_state.items() if n.startswith("k_pred.")}
v_state = {n.replace("v_pred.", "proj."): vv for n, vv in joint_state.items() if n.startswith("v_pred.")}
k_head.load_state_dict(k_state)
v_head.load_state_dict(v_state)
print("  K, V heads warm-started from kv_medusa_head_joint_one_1.pt")

# Q from previous kvq_optimal training
q_state_path = CKPT_DIR / "q_head_kvq_optimal.pt"
if q_state_path.exists():
    q_head.load_state_dict(torch.load(q_state_path, map_location=device))
    print(f"  Q head warm-started from {q_state_path.name}")
else:
    print("  Q head: fresh init")

decoder = WhitenedKVQDecoder(d_kv, d_q, d_model).to(device).to(torch.float32)
# Set whitening buffers
decoder.W_K = W_K.to(device).float()
decoder.W_V = W_V.to(device).float()
decoder.W_Q = W_Q.to(device).float()
decoder.mu_K = mu_K.to(device).float()
decoder.mu_V = mu_V.to(device).float()
decoder.mu_Q = mu_Q.to(device).float()

dec_state_path = CKPT_DIR / "decoder_kvq_optimal.pt"
if dec_state_path.exists():
    sd = torch.load(dec_state_path, map_location=device)
    # Only load proj weight (whitening is fresh)
    decoder.proj.load_state_dict({"weight": sd["proj.weight"]})
    print(f"  Decoder proj warm-started from {dec_state_path.name}")
else:
    print("  Decoder: fresh init")

opt = torch.optim.AdamW([
    {"params": k_head.parameters(), "lr": LR},
    {"params": v_head.parameters(), "lr": LR},
    {"params": q_head.parameters(), "lr": LR},
    {"params": decoder.proj.parameters(), "lr": LR},
], weight_decay=0.01)

print(f"\n{'='*60}")
print(f"WHITENED K+V+Q — train {TRAIN_STEPS} steps, joint MSE+CE")
print(f"{'='*60}\n")

k_head.train(); v_head.train(); q_head.train(); decoder.train()
step = 0
history = []

for batch in epoch_iter(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    captured_Q.clear()
    with torch.no_grad():
        out = model(inp, use_cache=True, output_hidden_states=True)
        h_l14 = out.hidden_states[INPUT_LAYER_KV].float()
        h_l15 = out.hidden_states[INPUT_LAYER_Q].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()
        actual_q = captured_Q["q"].float()

    h_in_kv = h_l14[:, :-OFFSET]
    h_in_q  = h_l15[:, :-OFFSET]
    target_k = actual_k[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_v = actual_v[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_q = actual_q[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_toks = inp[:, OFFSET:]

    ml = min(h_in_kv.shape[1], h_in_q.shape[1], target_k.shape[1], target_q.shape[1], target_toks.shape[1])
    h_in_kv, h_in_q = h_in_kv[:, :ml], h_in_q[:, :ml]
    target_k, target_v, target_q = target_k[:, :ml], target_v[:, :ml], target_q[:, :ml]
    target_toks = target_toks[:, :ml]

    pred_k = k_head(h_in_kv)
    pred_v = v_head(h_in_kv)
    pred_q = q_head(h_in_q)

    loss_mse = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v) + F.mse_loss(pred_q, target_q)

    pred_k_flat = pred_k.reshape(1, ml, d_kv)
    pred_v_flat = pred_v.reshape(1, ml, d_kv)
    pred_q_flat = pred_q.reshape(1, ml, d_q)
    logits = decoder(pred_k_flat, pred_v_flat, pred_q_flat, lm_head_weight)
    loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))

    loss = loss_mse + loss_ce
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(k_head.parameters()) + list(v_head.parameters()) +
        list(q_head.parameters()) + list(decoder.proj.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
            cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean().item()
            cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim), target_v.reshape(-1, head_dim), dim=-1).mean().item()
            cos_q = F.cosine_similarity(pred_q.reshape(-1, head_dim), target_q.reshape(-1, head_dim), dim=-1).mean().item()
        print(f"  step {step:>4}: loss={loss.item():.3f} mse={loss_mse.item():.3f} ce={loss_ce.item():.3f} "
              f"cos_k={cos_k:.3f} cos_v={cos_v:.3f} cos_q={cos_q:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({"step": step,
                        "mse": round(loss_mse.item(), 4),
                        "ce": round(loss_ce.item(), 4),
                        "cos_k": round(cos_k, 4),
                        "cos_v": round(cos_v, 4),
                        "cos_q": round(cos_q, 4),
                        "acc": round(acc, 4)})

torch.save(k_head.state_dict(), CKPT_DIR / "k_head_kvq_whitened.pt")
torch.save(v_head.state_dict(), CKPT_DIR / "v_head_kvq_whitened.pt")
torch.save(q_head.state_dict(), CKPT_DIR / "q_head_kvq_whitened.pt")
torch.save({"proj_weight": decoder.proj.weight.data,
            "W_K": decoder.W_K, "W_V": decoder.W_V, "W_Q": decoder.W_Q,
            "mu_K": decoder.mu_K, "mu_V": decoder.mu_V, "mu_Q": decoder.mu_Q},
           CKPT_DIR / "decoder_kvq_whitened.pt")

# ─── Eval ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nEVAL — offset {OFFSET}\n{'='*60}")
k_head.eval(); v_head.eval(); q_head.eval(); decoder.eval()
match_top1 = match_top5 = total = 0

for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    captured_Q.clear()
    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_l14 = out.hidden_states[INPUT_LAYER_KV].float()
        h_l15 = out.hidden_states[INPUT_LAYER_Q].float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        with torch.no_grad():
            pk = k_head(h_l14[:, t:t+1])
            pv = v_head(h_l14[:, t:t+1])
            pq = q_head(h_l15[:, t:t+1])
            pk_flat = pk.reshape(1, 1, d_kv)
            pv_flat = pv.reshape(1, 1, d_kv)
            pq_flat = pq.reshape(1, 1, d_q)
            logits = decoder(pk_flat, pv_flat, pq_flat, lm_head_weight)
            top1 = logits.argmax(-1).item()
            top5 = set(logits.topk(5, dim=-1).indices[0, 0].tolist())
        true_tok = baseline_toks[t + OFFSET].item()
        if top1 == true_tok: match_top1 += 1
        if true_tok in top5: match_top5 += 1
        total += 1

attn_layer.forward = orig_forward

t1 = match_top1 / total
t5 = match_top5 / total

print(f"\n{'='*70}")
print(f"K+V+Q WHITENED + WARM-START — offset {OFFSET}, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  top-1: {t1:.3f}   top-5: {t5:.3f}   (n={total})")
print(f"  Reference: K-only joint (1×1) — 0.21 / 0.46")
print(f"  Reference: K+V+Q optimal (no whitening) — 0.19 / 0.34")
print(f"  Reference: real-Q ceiling — 0.615 / 0.785")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "input_layer_kv": INPUT_LAYER_KV, "input_layer_q": INPUT_LAYER_Q,
               "offset": OFFSET, "top1": round(t1, 4), "top5": round(t5, 4),
               "n": total, "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
