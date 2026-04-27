"""Eagle drafter — autoregressive draft test on Qwen3-0.6B.

For each anchor t:
  Step 1: drafter(h_t, embed(tok_t))         -> h_{t+1}_pred -> tok_{t+1}_pred = argmax LM(h_{t+1}_pred)
  Step 2: drafter(h_{t+1}_pred, embed(tok_{t+1}_pred)) -> h_{t+2}_pred -> tok_{t+2}_pred
  ...

Compare per-offset token accuracy to:
  - Standard Medusa heads (parallel-from-h_t):  ~32% / 5% / 2.5% / 2.5% / 2%
  - The 'oracle' KV-Medusa ceiling: 96% per-position when tokens are right.

Also runs a verification forward pass with the drafted tokens — measures the
joint speculative-decode acceptance: probability that the model's verify top-1
matches the drafter's token at each offset.
"""
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
N_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
N_OFFSETS = 5
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_eagle_test_06b.json")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(self.weight.dtype)


class EagleDrafter(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_mult=4):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.head_dim = head_dim
        self.fc_in = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        ffn_dim = ffn_mult * d_model
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, h_prev, tok_embeds):
        x = self.fc_in(torch.cat([h_prev, tok_embeds], dim=-1))
        B, S, _ = x.shape
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device, dtype=scores.dtype), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ V).transpose(1, 2).reshape(B, S, -1)
        x = x + self.o_proj(attn_out)
        x_norm2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(x_norm2)) * self.up(x_norm2))
        return x


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
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
embed_layer = model.model.embed_tokens
lm_head_weight = model.lm_head.weight.detach()

print("Loading val tokens...")
val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

print("Loading drafter...")
drafter = EagleDrafter(d_model, n_attn_heads, head_dim).to(device)
drafter.load_state_dict(torch.load(CKPT_DIR / "eagle_drafter.pt", map_location=device))
drafter.eval()

# ─── Run autoregressive draft + verify ─────────────────────────────────────
raw_match = {k: 0 for k in range(1, N_OFFSETS + 1)}      # drafter draft == baseline top1
verify_accept = {k: 0 for k in range(1, N_OFFSETS + 1)}  # verify pass top1 == drafter draft
total = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=False)
        baseline_top1 = out.logits.argmax(-1)  # [1, seq] — natural top-1
        h_final = out.hidden_states[-1].float()  # [1, seq, d]

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue

        # Autoregressive drafting from anchor t
        # State at step k: predicted h at position t+k, drafted tok at position t+k
        h_curr = h_final[:, t:t+1]  # h_t (true)
        tok_curr = inp[:, t:t+1]    # tok_t (true)

        drafted_toks = []
        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                te = embed_layer(tok_curr).detach().float()
                # drafter takes [h_{p-1}, embed(tok_p)] and predicts h_p
                # input: (h_curr=h at pos t+k-1, te=embed of tok at t+k-1) -> predict h at t+k
                pred_h = drafter(h_curr, te)  # [1, 1, d]
                pred_logits = F.linear(pred_h.to(lm_head_weight.dtype), lm_head_weight).float()
                next_tok = pred_logits.argmax(-1)  # [1, 1]
            drafted_toks.append(next_tok.item())
            h_curr = pred_h
            tok_curr = next_tok

        # Raw accuracy: drafted tokens vs baseline natural greedy continuation
        for k in range(1, N_OFFSETS + 1):
            pos = t + k
            if pos >= SEQ_LEN: continue
            if drafted_toks[k - 1] == baseline_top1[0, pos - 1].item():
                raw_match[k] += 1

        # Verify: feed drafts back through model, check if verify top1 agrees
        verify_input = inp.clone()
        for k in range(1, N_OFFSETS + 1):
            if t + k < SEQ_LEN:
                verify_input[0, t + k] = drafted_toks[k - 1]

        with torch.no_grad():
            verify_top1 = model(verify_input, use_cache=False).logits.argmax(-1)

        for k in range(1, N_OFFSETS + 1):
            pos = t + k
            if pos >= SEQ_LEN: continue
            d_k = drafted_toks[k - 1]
            if verify_top1[0, pos - 1].item() == d_k:
                verify_accept[k] += 1
            total[k] += 1

    n_done += 1
    print(f"  seq {n_done}/{N_SEQS} done", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("EAGLE DRAFTER — autoregressive draft test, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  N seqs: {N_SEQS}, anchors per seq: {len(ANCHORS)}")
print(f"\n  {'offset':<8}{'raw acc':<14}{'verify accept':<18}")

results = []
for k in range(1, N_OFFSETS + 1):
    n = total[k]
    raw = raw_match[k] / n if n else 0
    va = verify_accept[k] / n if n else 0
    print(f"  t+{k:<6}{raw:<14.3f}{va:<18.3f}")
    results.append({"offset": k, "n": n,
                    "raw_acc": round(raw, 4),
                    "verify_accept": round(va, 4)})

def chain(rates):
    out = 1.0; prod = 1.0
    for r in rates:
        prod *= r; out += prod
    return out

raw_chain = chain([r["raw_acc"] for r in results])
va_chain = chain([r["verify_accept"] for r in results])
print(f"\n  Chained tokens/step:")
print(f"    Raw drafter accuracy:   {raw_chain:.3f}")
print(f"    Verify-accept (real):   {va_chain:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_seqs": N_SEQS, "anchors_per_seq": len(ANCHORS),
               "results": results,
               "chained_raw": round(raw_chain, 4),
               "chained_verify": round(va_chain, 4)}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
