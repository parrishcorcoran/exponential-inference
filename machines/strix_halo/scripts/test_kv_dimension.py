"""Test: does the KV coordinate extend the manifold dimension?

Hypothesis: KV adds a coordinate to the manifold with each token.
- Token 1: manifold = embedding only (~10D spatial)
- Token N: manifold = embedding + KV from N-1 tokens (more dimensions)

Test: measure intrinsic dimension of hidden states as a function of
sequence position. Collect hidden states at position P from MANY
different sequences, then estimate intrinsic dimension of that set.

If KV adds coordinates: intrinsic dim increases with position.
If KV doesn't add coordinates: intrinsic dim stays ~10 regardless.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json

device = "cuda"

print("=" * 70)
print("KV COORDINATE TEST: does intrinsic dimension grow with position?")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

HIDDEN = model.config.hidden_size
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Diverse prompts — need MANY sequences to measure dimension at each position
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Dolphins are highly intelligent marine mammals known for their social behavior and complex communication systems.",
    "The process of fermentation has been used for thousands of years to produce bread, wine, and cheese.",
    "Modern architecture emphasizes clean lines, open spaces, and the integration of buildings with their natural surroundings.",
    "The immune system protects the body against infection through a complex network of cells and proteins.",
    "Jazz music originated in the African American communities of New Orleans in the late nineteenth century.",
    "Volcanic eruptions can have devastating effects on local ecosystems but also create fertile soil over time.",
    "The printing press, invented by Gutenberg around 1440, revolutionized the spread of knowledge throughout Europe.",
    "Coral reefs are among the most diverse ecosystems on the planet, supporting thousands of marine species.",
    "The development of antibiotics in the twentieth century transformed medicine and saved millions of lives worldwide.",
    "Ancient Egyptian civilization flourished along the Nile River for over three thousand years before the common era.",
    "Artificial neural networks were inspired by the structure and function of biological brains in living organisms.",
    "The theory of plate tectonics explains how continents drift and why earthquakes occur along fault lines.",
    "Renaissance art was characterized by a renewed interest in classical antiquity and a focus on humanism.",
    "Black holes form when massive stars collapse under their own gravity at the end of their life cycle.",
    "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.",
    "Shakespeare wrote approximately thirty seven plays and one hundred fifty four sonnets during his literary career.",
    "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials like silicon crystals.",
    "The Amazon River is the largest river by discharge volume of water in the world flowing through South America.",
    "Genetic algorithms are optimization techniques inspired by the process of natural selection in biological evolution.",
    "The Great Wall of China was built over many centuries to protect the northern borders from various nomadic groups.",
    "Superconductors carry electrical current with zero resistance when cooled below their critical temperature point.",
    "The stock market operates as a platform where buyers and sellers trade shares of publicly listed companies.",
    "Glaciers form over many years from the accumulation and compaction of snow into dense ice masses.",
    "Machine learning models can be broadly categorized into supervised, unsupervised, and reinforcement learning approaches.",
    "The human brain contains approximately eighty six billion neurons connected by trillions of synapses.",
    "Coffee originated in Ethiopia and became one of the most widely consumed beverages across the entire globe.",
    "Quantum entanglement occurs when pairs of particles become correlated in ways that cannot be explained classically.",
    "The periodic table was first organized by Mendeleev based on atomic weight and recurring chemical properties.",
    "Earthquakes generate seismic waves that travel through the interior of the Earth and along its surface.",
    "Democracy originated in ancient Athens where citizens participated directly in making laws and political decisions.",
    "The Hubble Space Telescope has provided stunning images and data about distant galaxies and cosmic phenomena.",
    "Cryptographic hash functions map data of arbitrary size to fixed size values used for data integrity.",
    "Tropical rainforests receive more than two hundred centimeters of rainfall annually and maintain high biodiversity.",
    "The transistor, invented in 1947, became the fundamental building block of all modern electronic devices.",
    "Ocean currents distribute heat around the globe and significantly influence regional weather patterns and climate.",
    "Beethoven composed nine symphonies that are considered masterpieces of Western classical music tradition.",
    "Stem cells have the remarkable ability to develop into many different cell types in the body.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Urbanization has accelerated dramatically since the industrial revolution with more people living in cities than ever.",
]

print(f"Tokenizing {len(texts)} sequences...", flush=True)
all_ids = []
for text in texts:
    ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=64).input_ids[0]
    all_ids.append(ids)

# Pad to same length
max_len = max(len(x) for x in all_ids)
padded = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=device)
for i, ids in enumerate(all_ids):
    padded[i, :len(ids)] = ids.to(device)
lengths = [len(ids) for ids in all_ids]
min_len = min(lengths)
print(f"Sequences: {len(all_ids)}, min_len={min_len}, max_len={max_len}", flush=True)

# ═══════════════════════════════════════════════════════
# Collect hidden states at each position across all sequences
# ═══════════════════════════════════════════════════════
print(f"\nCollecting hidden states...", flush=True)

# We'll measure at several layers to see if the effect is layer-dependent
measure_layers = [0, 1, 5, 10, 20, 30, 39, 40]  # 0=embedding, 40=final

# For each (layer, position): collect hidden states from all sequences
# Then measure intrinsic dimension of that set
positions_to_test = [1, 2, 3, 5, 8, 12, 16, 20]
positions_to_test = [p for p in positions_to_test if p < min_len]

# Process in small batches to avoid OOM
BATCH = 4
all_hidden = {}  # (layer_idx, position) → list of hidden vectors

for batch_start in range(0, len(padded), BATCH):
    batch_ids = padded[batch_start:batch_start+BATCH]
    batch_lens = lengths[batch_start:batch_start+BATCH]

    with torch.no_grad():
        out = model.model(batch_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # (L+1) × [B, T, H]

        for layer_idx in measure_layers:
            if layer_idx >= len(hidden_states):
                continue
            for pos in positions_to_test:
                key = (layer_idx, pos)
                if key not in all_hidden:
                    all_hidden[key] = []
                for b in range(len(batch_ids)):
                    if pos < batch_lens[b]:
                        all_hidden[key].append(
                            hidden_states[layer_idx][b, pos].cpu().float().numpy()
                        )

    if (batch_start + BATCH) % 20 == 0:
        print(f"  {batch_start+BATCH}/{len(padded)} sequences", flush=True)

print(f"Collected hidden states for {len(all_hidden)} (layer, position) pairs", flush=True)

# ═══════════════════════════════════════════════════════
# TwoNN intrinsic dimension estimation
# ═══════════════════════════════════════════════════════

def twonn_dimension(X):
    """TwoNN estimator (Facco et al. 2017).
    X: numpy array [N, D]
    Returns estimated intrinsic dimension.
    """
    from scipy.spatial.distance import cdist
    N = X.shape[0]
    if N < 10:
        return float('nan')

    # Compute pairwise distances
    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)

    # For each point, find r1 (nearest) and r2 (second nearest)
    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    # Ratio mu = r2/r1
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]

    if len(mu) < 5:
        return float('nan')

    # Sort mu values
    mu_sorted = np.sort(mu)
    N_valid = len(mu_sorted)

    # Empirical CDF
    F_emp = np.arange(1, N_valid + 1) / N_valid

    # MLE: d = N / sum(log(mu_i))
    log_mu = np.log(mu_sorted)
    d_mle = N_valid / np.sum(log_mu)

    return d_mle


print(f"\n{'='*60}")
print("INTRINSIC DIMENSION vs SEQUENCE POSITION")
print(f"{'='*60}")

print(f"\n{'Layer':>6}", end="")
for pos in positions_to_test:
    print(f"  pos={pos:>2}", end="")
print()
print("-" * (8 + 8 * len(positions_to_test)))

results = {}
for layer_idx in measure_layers:
    dims = []
    print(f"  L={layer_idx:>2} ", end="")
    for pos in positions_to_test:
        key = (layer_idx, pos)
        if key in all_hidden and len(all_hidden[key]) >= 10:
            X = np.array(all_hidden[key])
            d = twonn_dimension(X)
            dims.append(d)
            print(f"  {d:>6.2f}", end="")
        else:
            dims.append(float('nan'))
            print(f"  {'N/A':>6}", end="")
    print()
    results[f"layer_{layer_idx}"] = {
        "positions": positions_to_test,
        "dimensions": dims
    }

# Also measure: KV cache rank growth
print(f"\n{'='*60}")
print("KV CACHE EFFECTIVE RANK vs POSITION")
print("If KV adds coordinates, rank should grow with position")
print(f"{'='*60}")

# Take one sequence, measure the effective rank of the KV cache at each position
test_ids = padded[0:1, :min_len]
with torch.no_grad():
    out = model(test_ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values

    print(f"\n{'Layer':>6} {'Seq len':>8} {'K rank':>8} {'V rank':>8} {'K fro':>10} {'V fro':>10}")
    print("-" * 55)

    for layer_idx in [0, 5, 10, 20, 30, 39]:
        k = past.layers[layer_idx].keys[0].float()   # [N_KV, T, HD]
        v = past.layers[layer_idx].values[0].float()

        # Reshape: treat all KV heads together [T, N_KV * HD]
        k_flat = k.permute(1, 0, 2).reshape(k.shape[1], -1)  # [T, N_KV*HD]
        v_flat = v.permute(1, 0, 2).reshape(v.shape[1], -1)

        # Effective rank via singular values
        k_svd = torch.linalg.svdvals(k_flat)
        v_svd = torch.linalg.svdvals(v_flat)

        # Effective rank: exp(entropy of normalized singular values)
        k_norm = k_svd / k_svd.sum()
        k_ent = -(k_norm * (k_norm + 1e-10).log()).sum()
        k_rank = torch.exp(k_ent).item()

        v_norm = v_svd / v_svd.sum()
        v_ent = -(v_norm * (v_norm + 1e-10).log()).sum()
        v_rank = torch.exp(v_ent).item()

        print(f"  L={layer_idx:>2}  {k_flat.shape[0]:>6}  {k_rank:>7.1f}  {v_rank:>7.1f}"
              f"  {k_flat.norm():>9.1f}  {v_flat.norm():>9.1f}")

# Now measure rank at different sequence lengths (truncate the KV)
print(f"\n{'='*60}")
print("KV RANK GROWTH: same layer, increasing sequence length")
print(f"{'='*60}")

layer_test = 20  # middle layer
k_full = past.layers[layer_test].keys[0].float()   # [N_KV, T, HD]
v_full = past.layers[layer_test].values[0].float()

print(f"\nLayer {layer_test}, measuring KV rank at different prefix lengths:")
print(f"{'Prefix':>8} {'K rank':>8} {'V rank':>8} {'K/prefix':>10}")
print("-" * 40)

for prefix_len in [2, 4, 6, 8, 12, 16, 20, min_len]:
    if prefix_len > k_full.shape[1]:
        break
    k_trunc = k_full[:, :prefix_len, :].permute(1, 0, 2).reshape(prefix_len, -1)
    v_trunc = v_full[:, :prefix_len, :].permute(1, 0, 2).reshape(prefix_len, -1)

    k_svd = torch.linalg.svdvals(k_trunc)
    v_svd = torch.linalg.svdvals(v_trunc)

    k_norm = k_svd / k_svd.sum()
    k_ent = -(k_norm * (k_norm + 1e-10).log()).sum()
    k_rank = torch.exp(k_ent).item()

    v_norm = v_svd / v_svd.sum()
    v_ent = -(v_norm * (v_norm + 1e-10).log()).sum()
    v_rank = torch.exp(v_ent).item()

    print(f"  {prefix_len:>6}  {k_rank:>7.1f}  {v_rank:>7.1f}  {k_rank/prefix_len:>9.3f}")

# Save
with open("machines/strix_halo/results/kv_dimension.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results.", flush=True)
