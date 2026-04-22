"""Gradient Axis Decomposition + Bimodal Lens.

1. Forward → backward → gradient on hidden states at each layer
2. Decompose gradient into principal directions (SVD)
3. Check: are gradient directions bimodal (carry at 0, flip at π)?
4. Does the gradient flow along the manifold's two-mode structure?

If yes: the gradient IS the manifold geometry decomposed into
carry (preserve) and flip (transform) channels.
"""
import torch
import torch.nn.functional as F
import numpy as np

device = "cuda"

print("=" * 70)
print("GRADIENT AXES + BIMODAL LENS")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device)
model.eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers

text = "The theory of general relativity describes gravity as the curvature of spacetime caused by mass"
ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
T = ids.shape[1]

print(f"Sequence: {T} tokens, H={H}, L={N_LAYERS}")

# ═══════════════════════════════════════════════════════
# Get per-layer gradients
# ═══════════════════════════════════════════════════════
print(f"\nCollecting per-layer gradients...", flush=True)

# Hook to capture intermediate hidden states with grad
layer_hiddens = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h.retain_grad()
        layer_hiddens[layer_idx] = h
    return hook

# Register hooks
hooks = []
for i in range(N_LAYERS):
    h = model.model.layers[i].register_forward_hook(make_hook(i))
    hooks.append(h)

# Forward
model.zero_grad()
out = model(ids)
logits = out.logits[0, -1]  # last position

# Target: max logit (the predicted token)
loss = logits.max()
loss.backward()

# Collect gradients
print(f"\nPer-layer gradient analysis:")
print(f"{'Layer':>6} {'Grad norm':>10} {'Carry%':>8} {'Flip%':>8} {'Top SV':>8} {'SV ratio':>9}")
print("-" * 55)

layer_grad_data = []

for i in range(N_LAYERS):
    if i not in layer_hiddens or layer_hiddens[i].grad is None:
        continue

    grad = layer_hiddens[i].grad[0, -1].float().cpu().numpy()  # [H] at last position
    h_val = layer_hiddens[i][0, -1].float().detach().cpu().numpy()

    grad_norm = np.linalg.norm(grad)

    # ── Bimodal decomposition ──
    # The rotation from layer i to layer i+1:
    # carry = component of grad aligned with hidden state (preserving)
    # flip = component perpendicular to hidden state (transforming)
    h_unit = h_val / (np.linalg.norm(h_val) + 1e-10)
    carry_component = np.dot(grad, h_unit)  # scalar projection
    carry_vec = carry_component * h_unit
    flip_vec = grad - carry_vec

    carry_frac = np.linalg.norm(carry_vec) / (grad_norm + 1e-10)
    flip_frac = np.linalg.norm(flip_vec) / (grad_norm + 1e-10)

    # ── SVD of gradient over all positions ──
    # Full gradient matrix [T, H] at this layer
    full_grad = layer_hiddens[i].grad[0].float().cpu().numpy()  # [T, H]
    U, S, Vt = np.linalg.svd(full_grad, full_matrices=False)

    # Top singular value and ratio (how concentrated is the gradient?)
    top_sv = S[0]
    sv_ratio = S[0] / (S[1] + 1e-10) if len(S) > 1 else float('inf')

    # Angle between top gradient direction and hidden state
    top_dir = Vt[0]  # top gradient direction
    angle = np.arccos(np.clip(np.abs(np.dot(top_dir, h_unit)), 0, 1))
    angle_deg = np.degrees(angle)

    layer_grad_data.append({
        "layer": i,
        "grad_norm": grad_norm,
        "carry_frac": carry_frac,
        "flip_frac": flip_frac,
        "top_sv": top_sv,
        "sv_ratio": sv_ratio,
        "angle_deg": angle_deg,
    })

    print(f"  L{i:>3} {grad_norm:>10.2f} {carry_frac:>7.1%} {flip_frac:>7.1%} "
          f"{top_sv:>8.2f} {sv_ratio:>8.1f}x")

# Clean up hooks
for h in hooks:
    h.remove()

# ═══════════════════════════════════════════════════════
# Analysis: is the gradient bimodal?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BIMODAL ANALYSIS")
print(f"{'='*60}")

carry_fracs = [d["carry_frac"] for d in layer_grad_data]
flip_fracs = [d["flip_frac"] for d in layer_grad_data]
angles = [d["angle_deg"] for d in layer_grad_data]
sv_ratios = [d["sv_ratio"] for d in layer_grad_data]

print(f"\n  Carry fraction: mean={np.mean(carry_fracs):.3f} std={np.std(carry_fracs):.3f}")
print(f"  Flip fraction:  mean={np.mean(flip_fracs):.3f} std={np.std(flip_fracs):.3f}")
print(f"  SV ratio:       mean={np.mean(sv_ratios):.1f} std={np.std(sv_ratios):.1f}")

# Per-layer trajectory: does carry/flip change through depth?
print(f"\n  Carry/flip through depth:")
for d in layer_grad_data:
    carry_bar = "█" * int(d["carry_frac"] * 30)
    flip_bar = "░" * int(d["flip_frac"] * 30)
    print(f"    L{d['layer']:>2}: carry {carry_bar} flip {flip_bar} | angle={d['angle_deg']:.0f}°")

# Is there a bimodal distribution of angles?
print(f"\n  Angle distribution (gradient direction vs hidden state):")
angle_bins = [0] * 9  # 0-10, 10-20, ..., 80-90 degrees
for a in angles:
    bin_idx = min(8, int(a / 10))
    angle_bins[bin_idx] += 1

for i, count in enumerate(angle_bins):
    bar = "█" * (count * 3)
    print(f"    {i*10:>2}°-{(i+1)*10:>2}°: {count:>3} {bar}")

# Early vs late layer comparison
mid = len(layer_grad_data) // 2
early_carry = np.mean([d["carry_frac"] for d in layer_grad_data[:mid]])
late_carry = np.mean([d["carry_frac"] for d in layer_grad_data[mid:]])
print(f"\n  Early layers (0-{mid}) avg carry: {early_carry:.3f}")
print(f"  Late layers ({mid}-{N_LAYERS}) avg carry:  {late_carry:.3f}")
print(f"  Shift: {'more carry late' if late_carry > early_carry else 'more flip late'}")

# ═══════════════════════════════════════════════════════
# Per-KV-position gradient (which context positions matter per axis)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GRADIENT PER AXIS PER POSITION")
print(f"{'='*60}")

# At middle layer: decompose gradient into carry and flip per position
mid_layer = N_LAYERS // 2
if mid_layer in layer_hiddens and layer_hiddens[mid_layer].grad is not None:
    full_grad = layer_hiddens[mid_layer].grad[0].float().cpu().numpy()  # [T, H]
    h_vals = layer_hiddens[mid_layer][0].float().detach().cpu().numpy()  # [T, H]

    print(f"\n  Layer {mid_layer} gradient decomposed per position:")
    print(f"  {'Pos':>4} {'Token':>12} {'Carry':>8} {'Flip':>8} {'Total':>8} {'%Carry':>7}")

    for t in range(T):
        g = full_grad[t]
        hv = h_vals[t]
        h_unit = hv / (np.linalg.norm(hv) + 1e-10)

        carry_comp = np.dot(g, h_unit)
        carry_v = carry_comp * h_unit
        flip_v = g - carry_v

        carry_n = np.linalg.norm(carry_v)
        flip_n = np.linalg.norm(flip_v)
        total_n = np.linalg.norm(g)
        pct_carry = carry_n / (total_n + 1e-10)

        tok = tokenizer.decode(ids[0, t:t+1])
        print(f"  {t:>4} {tok:>12} {carry_n:>8.2f} {flip_n:>8.2f} {total_n:>8.2f} {pct_carry:>6.1%}")

print(f"\nDone.", flush=True)
