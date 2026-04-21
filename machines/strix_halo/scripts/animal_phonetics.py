"""Universal Animal Phonetic Encoder.

Takes any audio → extracts precise acoustic features → encodes as text.
The text IS the animal's written language. Measurable, precise, universal.

Features extracted:
- F0: fundamental frequency (pitch) contour
- Harmonics: number and ratio of overtones
- Duration: length of vocalization
- Amplitude: envelope shape (attack, sustain, decay)
- Spectral centroid: brightness
- Bandwidth: frequency spread
- Modulation: rate of frequency/amplitude changes
- Formants: resonance peaks (vowel-like qualities)
"""
import numpy as np
import librosa
import json
import os

print("=" * 70)
print("UNIVERSAL ANIMAL PHONETIC ENCODER")
print("=" * 70)

def extract_phonetics(audio, sr):
    """Extract precise acoustic features from audio.

    Returns a structured phonetic description as a dictionary.
    """
    phonetics = {}

    # Duration
    duration = len(audio) / sr
    phonetics['duration_ms'] = round(duration * 1000)

    # Fundamental frequency (pitch) using pyin
    fmax = min(10000, sr // 2 - 1)
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=fmax, sr=sr)
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 0:
        phonetics['f0_mean'] = round(float(np.mean(f0_valid)))
        phonetics['f0_min'] = round(float(np.min(f0_valid)))
        phonetics['f0_max'] = round(float(np.max(f0_valid)))
        phonetics['f0_range'] = round(float(np.max(f0_valid) - np.min(f0_valid)))
        # Pitch contour shape: rising, falling, flat, arc
        if len(f0_valid) > 3:
            first_third = np.mean(f0_valid[:len(f0_valid)//3])
            last_third = np.mean(f0_valid[-len(f0_valid)//3:])
            mid_third = np.mean(f0_valid[len(f0_valid)//3:2*len(f0_valid)//3])
            if last_third > first_third * 1.1:
                phonetics['pitch_contour'] = 'rising'
            elif first_third > last_third * 1.1:
                phonetics['pitch_contour'] = 'falling'
            elif mid_third > max(first_third, last_third) * 1.05:
                phonetics['pitch_contour'] = 'arc'
            else:
                phonetics['pitch_contour'] = 'flat'
    else:
        phonetics['f0_mean'] = 0
        phonetics['pitch_contour'] = 'unvoiced'

    # Voiced fraction
    phonetics['voiced_pct'] = round(float(np.mean(voiced_flag)) * 100)

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    phonetics['centroid_mean'] = round(float(np.mean(centroid)))
    phonetics['centroid_std'] = round(float(np.std(centroid)))

    # Spectral bandwidth (frequency spread)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    phonetics['bandwidth_mean'] = round(float(np.mean(bandwidth)))

    # Spectral rolloff (high frequency content)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    phonetics['rolloff_mean'] = round(float(np.mean(rolloff)))

    # Harmonics: spectral flatness (tonal vs noisy)
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    phonetics['tonality'] = round(float(1.0 - np.mean(flatness)), 3)  # 1=tonal, 0=noise

    # Amplitude envelope
    rms = librosa.feature.rms(y=audio)[0]
    phonetics['loudness_mean'] = round(float(np.mean(rms)), 4)
    phonetics['loudness_max'] = round(float(np.max(rms)), 4)

    # Attack time (how fast the sound starts)
    if len(rms) > 3:
        peak_idx = np.argmax(rms)
        attack_time = peak_idx / len(rms)
        phonetics['attack'] = 'sudden' if attack_time < 0.15 else 'gradual' if attack_time < 0.4 else 'slow'

    # Modulation rate (how fast the sound changes)
    if len(centroid) > 5:
        mod = np.abs(np.diff(centroid))
        phonetics['modulation'] = 'fast' if np.mean(mod) > np.std(centroid) else 'slow'

    # MFCCs (vocal tract shape — like formants)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
    for i in range(5):
        phonetics[f'mfcc{i}'] = round(float(np.mean(mfccs[i])), 1)

    # Zero crossing rate (noisiness)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    phonetics['noisiness'] = round(float(np.mean(zcr)), 4)

    return phonetics


def phonetics_to_text(phonetics):
    """Convert phonetic features to a structured text string.

    This IS the animal's written language.
    """
    parts = []
    parts.append(f"DUR:{phonetics['duration_ms']}ms")
    parts.append(f"F0:{phonetics['f0_mean']}Hz")
    parts.append(f"CONTOUR:{phonetics.get('pitch_contour', 'unknown')}")
    parts.append(f"TONE:{phonetics['tonality']}")
    parts.append(f"BRIGHT:{phonetics['centroid_mean']}Hz")
    parts.append(f"BAND:{phonetics['bandwidth_mean']}Hz")
    parts.append(f"ATTACK:{phonetics.get('attack', 'unknown')}")
    parts.append(f"MOD:{phonetics.get('modulation', 'unknown')}")
    parts.append(f"VOICED:{phonetics['voiced_pct']}%")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════
# Test on whale data
# ═══════════════════════════════════════════════════════
print("\nTesting on whale recordings...")

from datasets import load_dataset
ds = load_dataset('confit/wmms-parquet', split='train')

results = []
species_phonetics = {}

for i in range(min(50, len(ds))):
    try:
        audio = ds[i]['audio']
        y = np.array(audio['array'], dtype=np.float32)
        sr = audio['sampling_rate']
        sp = ds[i]['species']

        if len(y) < sr * 0.1:
            continue

        phon = extract_phonetics(y, sr)
        text = phonetics_to_text(phon)

        results.append({'species': sp, 'phonetics': phon, 'text': text})

        if sp not in species_phonetics:
            species_phonetics[sp] = []
        species_phonetics[sp].append(phon)

    except Exception as e:
        pass

print(f"Processed: {len(results)} recordings")

# Show examples
print(f"\nEXAMPLES:")
for r in results[:5]:
    print(f"  [{r['species']}]")
    print(f"    {r['text']}")

# Compare species
print(f"\nSPECIES ACOUSTIC PROFILES:")
print(f"{'Species':>30} {'F0 Hz':>7} {'Bright':>7} {'Tonal':>6} {'Voiced':>7}")
print("-" * 62)

for sp, phons in sorted(species_phonetics.items()):
    if len(phons) < 2:
        continue
    f0s = [p['f0_mean'] for p in phons]
    brights = [p['centroid_mean'] for p in phons]
    tonals = [p['tonality'] for p in phons]
    voiceds = [p['voiced_pct'] for p in phons]

    print(f"{sp:>30} {np.mean(f0s):>6.0f} {np.mean(brights):>6.0f} "
          f"{np.mean(tonals):>6.2f} {np.mean(voiceds):>6.0f}%")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/data"
with open(f"{SAVE_DIR}/whale_phonetics.json", "w") as f:
    json.dump(results[:100], f, indent=2)
print(f"\nSaved to {SAVE_DIR}/whale_phonetics.json", flush=True)
