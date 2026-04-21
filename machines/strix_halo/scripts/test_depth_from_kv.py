"""Test: depth comes from KV + manifold, not manifold alone.

Prediction: more KV angles (longer context) = better manifold reconstruction
= more defined positions (higher top1 probability).

Test 1: Same text, but truncate context to different lengths.
        At each length, generate N tokens and measure what fraction
        are defined (top1 > 0.8). Longer context should = more defined.

Test 2: Same token in different KV contexts.
        The token "the" appears everywhere. Same manifold position.
        But with different KV (different preceding text), the next token
        differs. Depth comes from KV, not from the embedding alone.
"""
import torch
import torch.nn.functional as F
import json

device = "cuda"

print("=" * 70)
print("DEPTH FROM KV: does context length affect how defined tokens are?")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Test 1: Context length → defined fraction
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 1: Does longer context make more positions defined?")
print(f"{'='*60}")

# Long text that we'll truncate to different lengths
long_texts = [
    "The history of mathematics spans thousands of years and includes contributions from civilizations around the world. Ancient Egyptians used geometry for surveying land after the annual flooding of the Nile. Greek mathematicians like Euclid and Archimedes laid the foundations of formal proof. During the Islamic Golden Age, scholars preserved and extended Greek mathematics while developing algebra. The Renaissance brought new advances in calculus and analytic geometry through the work of Newton and Leibniz.",
    "Marine biology is the scientific study of organisms that live in the ocean and other saltwater environments. The ocean covers more than seventy percent of the Earth's surface and contains an incredible diversity of life forms. From microscopic plankton to enormous blue whales, marine ecosystems support complex food webs. Coral reefs are among the most biodiverse habitats on Earth, often called the rainforests of the sea. Deep sea hydrothermal vents support life through chemosynthesis rather than photosynthesis.",
    "The development of artificial intelligence has progressed through several distinct phases since its inception in the 1950s. Early systems relied on symbolic reasoning and hand-crafted rules to solve specific problems. The introduction of machine learning shifted the focus toward statistical pattern recognition from data. Deep learning emerged in the 2010s with neural networks achieving superhuman performance on many tasks. Modern large language models demonstrate remarkable abilities in text generation, reasoning, and following complex instructions.",
]

N_GEN = 20
context_lengths = [5, 10, 20, 40, 60, 80]

print(f"\n{'Ctx len':>8} {'Defined%':>9} {'Branch%':>9} {'Mid%':>7} {'Avg p1':>8} {'Avg entropy':>12}")
print("-" * 60)

results_test1 = []

for ctx_len in context_lengths:
    all_top1 = []
    all_entropy = []

    for text in long_texts:
        ids_full = tokenizer(text, return_tensors='pt').input_ids.to(device)
        if ids_full.shape[1] < ctx_len:
            continue
        ids = ids_full[:, :ctx_len]

        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values

            for step in range(N_GEN):
                if step == 0:
                    next_tok = out.logits[0, -1].argmax(-1)
                    probs = F.softmax(out.logits[0, -1].float(), dim=-1)
                else:
                    out = model(next_tok.view(1, 1), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    next_tok = out.logits[0, -1].argmax(-1)
                    probs = F.softmax(out.logits[0, -1].float(), dim=-1)

                top1 = probs.max().item()
                ent = -(probs * (probs + 1e-10).log()).sum().item()
                all_top1.append(top1)
                all_entropy.append(ent)

    if all_top1:
        top1_arr = torch.tensor(all_top1)
        defined_pct = (top1_arr > 0.8).float().mean().item() * 100
        branch_pct = (top1_arr < 0.3).float().mean().item() * 100
        mid_pct = 100 - defined_pct - branch_pct
        avg_p1 = top1_arr.mean().item()
        avg_ent = sum(all_entropy) / len(all_entropy)

        results_test1.append({
            "ctx_len": ctx_len, "defined_pct": defined_pct,
            "branch_pct": branch_pct, "avg_p1": avg_p1, "avg_ent": avg_ent
        })
        print(f"{ctx_len:>8} {defined_pct:>8.1f}% {branch_pct:>8.1f}% {mid_pct:>6.1f}% "
              f"{avg_p1:>8.3f} {avg_ent:>11.3f}")

# ═══════════════════════════════════════════════════════
# Test 2: Same token, different KV context → different depth
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 2: Same token 'the' in different KV contexts")
print("Same manifold position, different depth (KV) → different next token")
print(f"{'='*60}")

# Sentences where "the" appears at different points
contexts_with_the = [
    "I walked to the",
    "She looked at the",
    "The theory predicts the",
    "Scientists discovered the",
    "In the middle of the",
    "He reached for the",
    "The capital of the",
    "They measured the",
    "Running through the",
    "The first step is the",
    "Beyond the horizon lies the",
    "Understanding the nature of the",
]

print(f"\n{'Context':>40} {'Next token':>15} {'p1':>8} {'Entropy':>9}")
print("-" * 75)

the_next_tokens = []
the_p1s = []
the_entropies = []

for ctx in contexts_with_the:
    ids = tokenizer(ctx, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        out = model(ids)
        probs = F.softmax(out.logits[0, -1].float(), dim=-1)
        next_tok = probs.argmax().item()
        top1 = probs.max().item()
        ent = -(probs * (probs + 1e-10).log()).sum().item()

    next_str = tokenizer.decode([next_tok])
    the_next_tokens.append(next_str)
    the_p1s.append(top1)
    the_entropies.append(ent)

    print(f"{ctx:>40} {next_str:>15} {top1:>8.3f} {ent:>9.3f}")

unique_next = len(set(the_next_tokens))
print(f"\nSame manifold position ('the'), {len(contexts_with_the)} different KV contexts:")
print(f"  Unique next tokens: {unique_next}/{len(contexts_with_the)}")
print(f"  Average p1: {sum(the_p1s)/len(the_p1s):.3f}")
print(f"  p1 range: {min(the_p1s):.3f} — {max(the_p1s):.3f}")

# ═══════════════════════════════════════════════════════
# Test 3: KV rank growth correlates with definedness
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 3: KV rank at each step vs next-token definedness")
print("If depth (KV angles) makes tokens defined, rank should")
print("correlate with how defined the next position is")
print(f"{'='*60}")

text = long_texts[0]
ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=100).input_ids.to(device)

with torch.no_grad():
    out = model(ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values

    # Measure KV rank and definedness at each generation step
    print(f"\n{'Step':>5} {'Token':>12} {'KV rank':>8} {'p1':>8} {'Defined':>8}")
    print("-" * 50)

    step_data = []
    ref_layer = 20

    next_tok = out.logits[0, -1].argmax(-1)
    probs = F.softmax(out.logits[0, -1].float(), dim=-1)

    for step in range(25):
        # Measure KV rank
        k = past.layers[ref_layer].keys[0].float()  # [N_KV, T, HD]
        k_flat = k.permute(1, 0, 2).reshape(k.shape[1], -1)
        k_svd = torch.linalg.svdvals(k_flat)
        k_norm = k_svd / k_svd.sum()
        k_ent = -(k_norm * (k_norm + 1e-10).log()).sum()
        k_rank = torch.exp(k_ent).item()

        top1 = probs.max().item()
        tok_str = tokenizer.decode([next_tok.item()])
        defined = "DEFINED" if top1 > 0.8 else "branch" if top1 < 0.3 else ""

        step_data.append({"step": step, "rank": k_rank, "p1": top1})
        print(f"{step:>5} {tok_str:>12} {k_rank:>8.1f} {top1:>8.3f} {defined:>8}")

        # Next step
        out = model(next_tok.view(1, 1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1].argmax(-1)
        probs = F.softmax(out.logits[0, -1].float(), dim=-1)

    # Correlation
    import numpy as np
    ranks = np.array([d["rank"] for d in step_data])
    p1s = np.array([d["p1"] for d in step_data])
    if ranks.std() > 0 and p1s.std() > 0:
        corr = np.corrcoef(ranks, p1s)[0, 1]
        print(f"\nCorrelation(KV_rank, p1): r = {corr:+.3f}")

# Save
with open("machines/strix_halo/results/depth_from_kv.json", "w") as f:
    json.dump({"test1": results_test1, "unique_next_for_the": unique_next}, f, indent=2)
print(f"\nSaved results.", flush=True)
