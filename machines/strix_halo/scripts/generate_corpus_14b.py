"""Generate teacher corpus from Qwen3-14B on Strix Halo.

Produces corpus.pt: tokenized sequences from C4 for distillation.
Target: 100K+ tokens from diverse text.
"""
import torch, time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = 'cuda'
TARGET_TOKENS = 100_000
MAX_LEN = 256
OUT = Path("machines/strix_halo/scratch/corpora/corpus.pt")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading Qwen3-14B for corpus generation...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
# Don't need the model for corpus — just tokenize C4 text
print("Streaming C4 data...", flush=True)

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
sequences = []
total_tokens = 0
t0 = time.time()

for i, item in enumerate(ds):
    text = item.get('text', '')
    if len(text) < 200:
        continue
    ids = tokenizer(text[:4000], return_tensors='pt',
                    truncation=True, max_length=MAX_LEN).input_ids[0]
    if len(ids) >= 32:
        sequences.append(ids)
        total_tokens += len(ids)
    if total_tokens >= TARGET_TOKENS:
        break
    if (i + 1) % 200 == 0:
        print(f"  {len(sequences)} seqs, {total_tokens} tokens [{time.time()-t0:.0f}s]", flush=True)

print(f"\nCorpus: {len(sequences)} sequences, {total_tokens} tokens", flush=True)

torch.save({
    "sequences": sequences,
    "total_tokens": total_tokens,
    "max_len": MAX_LEN,
    "tokenizer": "Qwen/Qwen3-14B",
    "source": "allenai/c4",
}, OUT)
print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", flush=True)
