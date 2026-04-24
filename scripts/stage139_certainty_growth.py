"""
Stage 139 — Per-position certainty growth measurement.

The missing piece: as a sentence is produced, the model should become
increasingly certain. This measures it directly.

For each position t in a long sequence:
  - Output entropy: H_t = -Σ p_i log p_i over vocabulary
  - Top-1 confidence: max(p_i)
  - Attention concentration (Gini, mean across heads/layers, restricted to
    rows where t is the query)
  - Residual stream change magnitude |h_t - h_{t-1}| at final layer

Predicted: entropy drops sharply in first ~10 tokens, then decays slowly.
If true: per-position compression should scale inversely with entropy
(low entropy = high certainty = compressible aggressively).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def gini_row(p):
    """Gini coefficient for non-negative 1D array."""
    p = np.sort(np.asarray(p, dtype=np.float64))
    if p.sum() < 1e-10: return 0.0
    n = len(p)
    cumsum = p.cumsum()
    return float((n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n)


@torch.no_grad()
def measure_certainty(model, tokens, seq_len, device):
    """For one sequence, return per-position metrics."""
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=False, output_hidden_states=True,
                 output_attentions=True)
    logits = out.logits[0]  # [seq, vocab]

    # Per-position output stats
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # [seq]
    top1 = probs.max(dim=-1).values.cpu().numpy()  # [seq]

    # Per-position attention concentration (Gini)
    # Average across all layers, all heads, taking the query row at each position
    # attns[l]: [num_heads, seq, seq]
    attns = out.attentions
    L = len(attns)
    seq_actual = logits.shape[0]
    attn_gini = np.zeros(seq_actual)
    for l in range(L):
        A = attns[l][0].cpu().float().numpy()  # [num_heads, seq, seq]
        for t in range(seq_actual):
            ginis = []
            for h in range(A.shape[0]):
                row = A[h, t, :t+1]
                if row.sum() > 1e-10:
                    ginis.append(gini_row(row))
            if ginis:
                attn_gini[t] += np.mean(ginis)
    attn_gini /= L

    # Per-position residual change at final layer
    # hidden_states[L]: input to final norm, shape [1, seq, d]
    h_final = out.hidden_states[-1][0].float().cpu().numpy()  # [seq, d]
    res_change = np.zeros(seq_actual)
    for t in range(1, seq_actual):
        res_change[t] = float(np.linalg.norm(h_final[t] - h_final[t-1]))

    return {
        "entropy": entropy.tolist(),
        "top1": top1.tolist(),
        "attn_gini_mean": attn_gini.tolist(),
        "res_change": res_change.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage139_certainty.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-sequences", type=int, default=5,
                   help="average across N sequences")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()

    print(f"loading {args.n_sequences} sequences of {args.seq_len} tokens each...")
    all_tokens = load_tokens(tok, args.seq_len * args.n_sequences * 2, "train")

    all_runs = []
    for i in range(args.n_sequences):
        start = i * args.seq_len
        if start + args.seq_len > len(all_tokens): break
        chunk = all_tokens[start:start + args.seq_len]
        print(f"  sequence {i+1}/{args.n_sequences}...")
        t0 = time.time()
        run = measure_certainty(model, chunk, args.seq_len, device)
        print(f"    done in {time.time()-t0:.0f}s")
        all_runs.append(run)

    # Average across sequences
    n = len(all_runs)
    seq_actual = len(all_runs[0]["entropy"])
    avg = {k: np.mean([r[k] for r in all_runs], axis=0) for k in all_runs[0]}

    print(f"\n{'='*60}\n=== per-position averages ===\n{'='*60}")
    print(f"  pos | entropy | top1   | attn_gini | res_change")
    print(f"  ----|---------|--------|-----------|-----------")
    sample_positions = [0, 1, 2, 5, 10, 20, 50, 100, 150, 200, seq_actual-1]
    sample_positions = [t for t in sample_positions if t < seq_actual]
    for t in sample_positions:
        print(f"  {t:>3d} | {avg['entropy'][t]:>7.3f} | {avg['top1'][t]:>6.4f} | "
              f"{avg['attn_gini_mean'][t]:>9.4f} | {avg['res_change'][t]:>9.3f}")

    # Decay analysis
    print(f"\n=== decay analysis ===")
    early_window = slice(2, 10)
    late_window = slice(seq_actual - 50, seq_actual)
    e_early = avg['entropy'][early_window].mean()
    e_late = avg['entropy'][late_window].mean()
    t1_early = avg['top1'][early_window].mean()
    t1_late = avg['top1'][late_window].mean()
    g_early = avg['attn_gini_mean'][early_window].mean()
    g_late = avg['attn_gini_mean'][late_window].mean()
    print(f"  entropy   early={e_early:.3f}  late={e_late:.3f}  ratio={e_early/max(e_late,1e-10):.2f}")
    print(f"  top1      early={t1_early:.4f}  late={t1_late:.4f}  ratio={t1_late/max(t1_early,1e-10):.2f}")
    print(f"  attn_gini early={g_early:.4f}  late={g_late:.4f}  ratio={g_late/max(g_early,1e-10):.2f}")

    # Verdict
    if e_late < e_early * 0.8:
        verdict_e = "ENTROPY DROPS — model gets more certain"
    elif e_late > e_early * 1.2:
        verdict_e = "ENTROPY GROWS — model gets less certain (unusual)"
    else:
        verdict_e = "ENTROPY ROUGHLY FLAT"
    print(f"\n  verdict (entropy): {verdict_e}")

    # Save
    results = {
        "model": args.model,
        "seq_len": args.seq_len,
        "n_sequences": args.n_sequences,
        "per_position_avg": {k: v.tolist() for k, v in avg.items()},
        "early_window": [2, 10],
        "late_window": [seq_actual - 50, seq_actual],
        "entropy_early": float(e_early),
        "entropy_late": float(e_late),
        "top1_early": float(t1_early),
        "top1_late": float(t1_late),
        "gini_early": float(g_early),
        "gini_late": float(g_late),
        "verdict": verdict_e,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
