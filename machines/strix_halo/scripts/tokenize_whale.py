"""Tokenize marine mammal sounds: spectrogram → BPE → sequences.

Step 1: Convert audio to mel spectrograms
Step 2: Quantize spectrogram patches into discrete tokens
Step 3: Build BPE vocabulary from token sequences
Step 4: (Later) Train model + measure manifold dimension
"""
import torch
import numpy as np
import librosa
import json
import os
from collections import Counter

print("=" * 70)
print("WHALE SOUND TOKENIZER — spectrogram → tokens")
print("=" * 70)

from datasets import load_dataset

print("Loading Watkins Marine Mammal Sound Database...")
ds = load_dataset('confit/wmms-parquet', split='train')
print(f"Loaded: {len(ds)} recordings, 32 species")

# ═══════════════════════════════════════════════════════
# Step 1: Convert to mel spectrograms
# ═══════════════════════════════════════════════════════
print(f"\nStep 1: Computing mel spectrograms...")

N_MELS = 64        # mel frequency bins
HOP_LENGTH = 512   # ~32ms at 16kHz
N_FFT = 1024

all_specs = []
all_species = []
n_errors = 0

for i in range(len(ds)):
    try:
        audio = ds[i]['audio']
        y = np.array(audio['array'], dtype=np.float32)
        sr = audio['sampling_rate']

        # Skip very short clips
        if len(y) < sr * 0.5:  # < 0.5 seconds
            continue

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                             hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        all_specs.append(mel_db)
        all_species.append(ds[i]['species'])

    except Exception as e:
        n_errors += 1

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(ds)} processed, {len(all_specs)} valid, {n_errors} errors")

print(f"\nSpectrograms: {len(all_specs)} valid recordings")
print(f"Species distribution:")
species_counts = Counter(all_species)
for species, count in species_counts.most_common(10):
    print(f"  {species}: {count}")

# ═══════════════════════════════════════════════════════
# Step 2: Quantize spectrogram patches into discrete tokens
# Using K-means on spectrogram column vectors (time frames)
# ═══════════════════════════════════════════════════════
print(f"\nStep 2: Quantizing spectrogram frames into tokens...")

# Collect all frames
all_frames = []
for spec in all_specs:
    # Each column is one time frame [N_MELS]
    for t in range(spec.shape[1]):
        all_frames.append(spec[:, t])

all_frames = np.stack(all_frames)
print(f"Total frames: {all_frames.shape[0]}")

# K-means clustering to create vocabulary
from sklearn.cluster import MiniBatchKMeans

VOCAB_SIZE = 512  # number of discrete tokens
print(f"K-means clustering into {VOCAB_SIZE} tokens...")

# Subsample for K-means (use max 100K frames)
n_sample = min(100000, len(all_frames))
idx = np.random.choice(len(all_frames), n_sample, replace=False)
kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000, n_init=3, random_state=42)
kmeans.fit(all_frames[idx])

print(f"K-means done. Cluster sizes:")
labels_sample = kmeans.predict(all_frames[idx])
counts = np.bincount(labels_sample, minlength=VOCAB_SIZE)
print(f"  Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.0f}")

# ═══════════════════════════════════════════════════════
# Step 3: Convert all spectrograms to token sequences
# ═══════════════════════════════════════════════════════
print(f"\nStep 3: Converting spectrograms to token sequences...")

token_sequences = []
sequence_species = []

for i, spec in enumerate(all_specs):
    frames = spec.T  # [T, N_MELS]
    tokens = kmeans.predict(frames)
    token_sequences.append(tokens.tolist())
    sequence_species.append(all_species[i])

# Stats
seq_lens = [len(s) for s in token_sequences]
print(f"Sequences: {len(token_sequences)}")
print(f"Length: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")
print(f"Total tokens: {sum(seq_lens)}")

# Show example
print(f"\nExample sequence (first 30 tokens):")
print(f"  Species: {sequence_species[0]}")
print(f"  Tokens: {token_sequences[0][:30]}")

# Per-species stats
print(f"\nPer-species token statistics:")
for species in species_counts.most_common(5):
    sp = species[0]
    sp_seqs = [s for s, sp2 in zip(token_sequences, sequence_species) if sp2 == sp]
    sp_tokens = [t for s in sp_seqs for t in s]
    n_unique = len(set(sp_tokens))
    print(f"  {sp}: {len(sp_seqs)} seqs, {len(sp_tokens)} tokens, {n_unique}/{VOCAB_SIZE} unique")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/data"
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, "whale_tokens.json")
with open(save_path, "w") as f:
    json.dump({
        "vocab_size": VOCAB_SIZE,
        "n_mels": N_MELS,
        "n_sequences": len(token_sequences),
        "sequences": token_sequences[:100],  # save first 100 for now
        "species": sequence_species[:100],
    }, f)

# Save kmeans centers for later use
np.save(os.path.join(SAVE_DIR, "whale_kmeans_centers.npy"), kmeans.cluster_centers_)

print(f"\nSaved to {SAVE_DIR}")
print(f"Vocab: {VOCAB_SIZE} tokens from K-means on {N_MELS}-dim mel frames")
print(f"Ready for manifold measurement.", flush=True)
