"""Analyze Z8G4's cross-model manifold fingerprints.

Reads all fingerprint_*.json in machines/z8g4/results/ and reports:
- TwoNN grand mean per model (Finding 01 cross-family)
- Rotation-curve mean angle per model (Finding 02 cross-family)
- Carry/flip/mid fractions per model (candidate Finding 14 cross-family)
- Per-transition detail for select layers
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_fingerprints():
    results_dir = REPO_ROOT / "machines" / "z8g4" / "results"
    data = {}
    for p in sorted(results_dir.glob("fingerprint_*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            if "twonn_mean_per_layer" not in d:
                continue
            data[p.stem.replace("fingerprint_", "")] = d
        except Exception as e:
            print(f"skip {p}: {e}")
    return data


def main():
    data = load_fingerprints()
    print(f"loaded {len(data)} fingerprints: {', '.join(data.keys())}\n")

    # Grand summary table
    print(f"=== top-line per model ===")
    print(f"  {'model':>35}  {'L':>3}  {'TwoNN':>8}  "
          f"{'mean_rot':>9}  {'carry':>6}  {'flip':>5}  {'mid':>5}")
    rows = []
    for name, d in data.items():
        L = d.get("L", len(d["twonn_mean_per_layer"]) - 1)
        twonn_mean = d.get("twonn_mean_grand", float("nan"))
        rot = d.get("mean_rotation_angle_per_transition", [])
        rot_valid = [r for r in rot if r is not None]
        rot_mean = sum(rot_valid) / len(rot_valid) if rot_valid else float("nan")
        cf = d.get("carry_fraction_per_transition", [])
        ff = d.get("flip_fraction_per_transition", [])
        mc = d.get("mode_concentration_per_transition", [])
        carry_frac = sum(c for c in cf if c is not None) / len([c for c in cf if c is not None]) if cf else float("nan")
        flip_frac = sum(c for c in ff if c is not None) / len([c for c in ff if c is not None]) if ff else float("nan")
        mid_frac = 1.0 - carry_frac - flip_frac if carry_frac == carry_frac else float("nan")
        ft_overlap = d.get("first_to_last_carry_overlap", float("nan"))
        adj_overlap = d.get("adjacent_carry_overlap", [])
        adj_valid = [a for a in adj_overlap if a is not None]
        adj_mean = sum(adj_valid) / len(adj_valid) if adj_valid else float("nan")
        rows.append({
            "name": name, "L": L, "twonn": twonn_mean, "rot": rot_mean,
            "carry_frac": carry_frac, "flip_frac": flip_frac, "mid_frac": mid_frac,
            "ft_overlap": ft_overlap, "adj_overlap": adj_mean,
        })
        print(f"  {name:>35}  {L:>3}  {twonn_mean:>8.3f}  "
              f"{rot_mean:>9.3f}  {carry_frac:>6.3f}  {flip_frac:>5.3f}  {mid_frac:>5.3f}")

    # Tokenizer-family groupings
    qwen = [r for r in rows if "Qwen" in r["name"] or "qwen" in r["name"]]
    llama = [r for r in rows if "TinyLlama" in r["name"] or "Llama" in r["name"] or "llama" in r["name"]]
    mistral = [r for r in rows if "Mistral" in r["name"] or "mistral" in r["name"]]
    phi = [r for r in rows if "phi" in r["name"].lower()]

    print(f"\n=== grouped by tokenizer family ===")
    for family_name, group in [("Qwen", qwen), ("Llama", llama), ("Mistral", mistral), ("Phi", phi)]:
        if not group: continue
        t = torch.tensor([r["twonn"] for r in group], dtype=torch.float32)
        cf = torch.tensor([r["carry_frac"] for r in group], dtype=torch.float32)
        ff = torch.tensor([r["flip_frac"] for r in group], dtype=torch.float32)
        rot = torch.tensor([r["rot"] for r in group], dtype=torch.float32)
        print(f"\n  {family_name} ({len(group)} models):")
        print(f"    TwoNN:     mean={float(t.mean()):.3f}  std={float(t.std()):.3f}  "
              f"range=[{float(t.min()):.2f}, {float(t.max()):.2f}]")
        print(f"    carry frac: mean={float(cf.mean()):.3f}  std={float(cf.std()):.3f}")
        print(f"    flip frac:  mean={float(ff.mean()):.3f}  std={float(ff.std()):.3f}")
        print(f"    mean rot:   mean={float(rot.mean()):.3f}  std={float(rot.std()):.3f}")
        adj = torch.tensor([r["adj_overlap"] for r in group if r["adj_overlap"] == r["adj_overlap"]], dtype=torch.float32)
        ft = torch.tensor([r["ft_overlap"] for r in group if r["ft_overlap"] == r["ft_overlap"]], dtype=torch.float32)
        if len(adj) > 0:
            print(f"    adj carry overlap: mean={float(adj.mean()):.3f}")
        if len(ft) > 0:
            print(f"    first-to-last carry overlap: mean={float(ft.mean()):.3f}  range=[{float(ft.min()):.3f}, {float(ft.max()):.3f}]")

    # Cross-family comparison: is two-mode structure universal?
    print(f"\n=== universality check (two-mode structure) ===")
    all_carry = [r["carry_frac"] for r in rows]
    all_flip = [r["flip_frac"] for r in rows]
    all_twonn = [r["twonn"] for r in rows]
    t_c = torch.tensor(all_carry, dtype=torch.float32)
    t_f = torch.tensor(all_flip, dtype=torch.float32)
    t_t = torch.tensor(all_twonn, dtype=torch.float32)
    print(f"  across all {len(rows)} models (4 tokenizer families):")
    print(f"    TwoNN mean:  {float(t_t.mean()):.3f}  std={float(t_t.std()):.3f}")
    print(f"    carry frac:  {float(t_c.mean()):.3f}  std={float(t_c.std()):.3f}")
    print(f"    flip frac:   {float(t_f.mean()):.3f}  std={float(t_f.std()):.3f}")

    if float(t_t.std()) < 2.0:
        print(f"  TwoNN: all models within ±{float(t_t.std()):.1f} dims → CROSS-FAMILY UNIVERSAL")
    if float(t_c.std()) < 0.05:
        print(f"  carry fraction: tight (±{float(t_c.std()):.3f}) → two-mode structure is universal")
    elif float(t_c.std()) < 0.10:
        print(f"  carry fraction: somewhat tight (±{float(t_c.std()):.3f}) → likely universal")
    else:
        print(f"  carry fraction: varies materially (±{float(t_c.std()):.3f}) → family-specific?")


if __name__ == "__main__":
    main()
