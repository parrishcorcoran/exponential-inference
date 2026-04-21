"""Step 1: 128-head model with per-token activation + early exit.

Same Qwen3-14B weights, reshaped to 128 heads of 40 dims.
Distill against original teacher to recover quality.
Then: router activates N heads, exits at layer L. One token, fast.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
import random
import copy
device = 'cuda'

print("="*70, flush=True)
print("128-HEAD MODEL: train + route", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

# Load teacher (original 40 heads)
print("\n[1/4] Loading teacher (40 heads)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

H = teacher.config.hidden_size           # 5120
N_LAYERS = teacher.config.num_hidden_layers  # 40
OLD_HEADS = teacher.config.num_attention_heads  # 40
OLD_KV = teacher.config.num_key_value_heads    # 8
OLD_HD = H // OLD_HEADS                        # 128

NEW_HEADS = 128
NEW_HD = H // NEW_HEADS  # 40
NEW_KV = 32  # scale KV heads proportionally (8 × 128/40 ≈ 32... use 32)
# Actually keep KV heads same ratio: 128/40 * 8 = 25.6 → 32

print(f"  Teacher: {OLD_HEADS} heads × {OLD_HD} dim, {OLD_KV} KV", flush=True)
print(f"  Student: {NEW_HEADS} heads × {NEW_HD} dim, {NEW_KV} KV", flush=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# [2/4] Create student by modifying attention config
print("\n[2/4] Creating 128-head student...", flush=True)

# Deep copy and modify config
student_config = copy.deepcopy(teacher.config)
student_config.num_attention_heads = NEW_HEADS
student_config.num_key_value_heads = NEW_KV

# Can't easily re-init with different head count from pretrained.
# Instead: use the teacher model directly, but override the attention
# reshape in the forward pass. The Q/K/V/O weight matrices are the same
# size (5120×5120 for Q, etc). We just change how they split into heads.

student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to(device)

# Monkey-patch each attention layer to use 128 heads
for layer in student.model.layers:
    attn = layer.self_attn
    attn.num_heads = NEW_HEADS
    attn.head_dim = NEW_HD
    attn.num_key_value_heads = NEW_KV
    attn.num_key_value_groups = NEW_HEADS // NEW_KV

    # Resize K/V projections: original is 8*128=1024 → need 32*40=1280
    # The weight matrices need to change size for K/V
    # Q: 5120→5120 (same, just reshaped)
    # K: 5120→1024 (old) → need 5120→1280 (new)
    # V: same as K
    # O: 5120→5120 (same)

    old_k_w = attn.k_proj.weight.data  # [1024, 5120]
    old_v_w = attn.v_proj.weight.data  # [1024, 5120]

    # Expand K/V by repeating and reshaping
    # From 8 KV heads of 128 dim → 32 KV heads of 40 dim
    # Repeat each old KV head 4× then truncate dims
    # Actually simpler: initialize new K/V with random init, train via distillation
    new_k = nn.Linear(H, NEW_KV * NEW_HD, bias=False, dtype=torch.bfloat16).to(device)
    new_v = nn.Linear(H, NEW_KV * NEW_HD, bias=False, dtype=torch.bfloat16).to(device)

    # Initialize from old weights: tile and slice
    # old: [8, 128, 5120] → reshape each head's 128 dims into 4 sub-heads of 40 dims
    # This gives us 8*4=32 KV heads of 40 dims each
    old_k_reshaped = old_k_w.view(OLD_KV, OLD_HD, H)  # [8, 128, 5120]
    # Split each 128-dim head into 3.2 sub-heads of 40... not clean division
    # Just repeat: each old head → 4 new heads (take first 40 dims of each repeat)
    new_k_init = old_k_reshaped.repeat_interleave(NEW_KV // OLD_KV, dim=0)  # [32, 128, 5120]
    new_k_init = new_k_init[:, :NEW_HD, :]  # [32, 40, 5120]
    new_k.weight.data = new_k_init.reshape(NEW_KV * NEW_HD, H)

    new_v_init = old_v_w.view(OLD_KV, OLD_HD, H).repeat_interleave(NEW_KV // OLD_KV, dim=0)[:, :NEW_HD, :]
    new_v.weight.data = new_v_init.reshape(NEW_KV * NEW_HD, H)

    attn.k_proj = new_k
    attn.v_proj = new_v

    # Also need to fix q_norm and k_norm if they exist (they normalize per head_dim)
    if hasattr(attn, 'q_norm') and attn.q_norm is not None:
        attn.q_norm = nn.RMSNorm(NEW_HD, eps=1e-6).to(device).to(torch.bfloat16)
    if hasattr(attn, 'k_norm') and attn.k_norm is not None:
        attn.k_norm = nn.RMSNorm(NEW_HD, eps=1e-6).to(device).to(torch.bfloat16)

# Freeze everything except K/V projections + norms (the parts we changed)
for p in student.parameters():
    p.requires_grad_(False)
trainable = 0
for layer in student.model.layers:
    attn = layer.self_attn
    for p in attn.k_proj.parameters():
        p.requires_grad_(True); trainable += p.numel()
    for p in attn.v_proj.parameters():
        p.requires_grad_(True); trainable += p.numel()
    if attn.q_norm is not None:
        for p in attn.q_norm.parameters():
            p.requires_grad_(True); trainable += p.numel()
    if attn.k_norm is not None:
        for p in attn.k_norm.parameters():
            p.requires_grad_(True); trainable += p.numel()

print(f"  Trainable: {trainable/1e6:.1f}M params", flush=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# [3/4] Distill student against teacher
print("\n[3/4] Distilling (KL against teacher)...", flush=True)

from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
batches = []
total_tok = 0
for item in ds:
    if len(item['text']) < 200:
        continue
    ids = tokenizer(item['text'][:1000], return_tensors='pt',
                    truncation=True, max_length=128).input_ids
    if ids.shape[1] >= 16:
        batches.append(ids)
        total_tok += ids.shape[1]
    if total_tok >= 50000:
        break
print(f"  {len(batches)} batches, {total_tok} tokens", flush=True)

opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=3e-4)
student.train()
t0 = time.time()

for epoch in range(3):
    random.shuffle(batches)
    total_kl = 0; nb = 0
    for batch in batches:
        b = batch.to(device)
        with torch.no_grad():
            t_logits = teacher(b, use_cache=False).logits.detach()
        s_logits = student(b, use_cache=False).logits

        kl = F.kl_div(
            F.log_softmax(s_logits.float() / 2, dim=-1),
            F.softmax(t_logits.float() / 2, dim=-1),
            reduction="batchmean"
        ) * 4

        opt.zero_grad()
        kl.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 0.5)
        opt.step()
        total_kl += kl.item(); nb += 1

    print(f"  epoch {epoch+1}  kl={total_kl/nb:.4f}  [{time.time()-t0:.0f}s]", flush=True)

student.eval()

# [4/4] Test: generate with full 128 heads vs subset
print("\n[4/4] Testing generation...", flush=True)
test_ids = tokenizer("The future of AI will", return_tensors='pt').input_ids.to(device)

# Full student (all 128 heads)
with torch.no_grad():
    out = student.generate(test_ids, max_new_tokens=40, do_sample=False)
student_text = tokenizer.decode(out[0][test_ids.shape[1]:], skip_special_tokens=True)

# Teacher baseline
with torch.no_grad():
    out = teacher.generate(test_ids, max_new_tokens=40, do_sample=False)
teacher_text = tokenizer.decode(out[0][test_ids.shape[1]:], skip_special_tokens=True)

print(f"  Teacher (40 heads):  {teacher_text[:70]}", flush=True)
print(f"  Student (128 heads): {student_text[:70]}", flush=True)

# Speed comparison
with torch.no_grad(): student.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): student.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
student_tps = 64 / (time.time() - t0)

with torch.no_grad(): teacher.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): teacher.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
teacher_tps = 64 / (time.time() - t0)

print(f"\n  Teacher speed: {teacher_tps:.1f} tok/s", flush=True)
print(f"  Student speed: {student_tps:.1f} tok/s", flush=True)

# Save
results = {
    "teacher_heads": OLD_HEADS, "student_heads": NEW_HEADS,
    "teacher_tps": teacher_tps, "student_tps": student_tps,
    "trainable_M": trainable / 1e6,
    "teacher_text": teacher_text, "student_text": student_text,
}
with open("machines/strix_halo/results/128head_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved 128head_14b.json", flush=True)
print(f"Step 1 complete. Next: route N of 128 heads per token.", flush=True)
