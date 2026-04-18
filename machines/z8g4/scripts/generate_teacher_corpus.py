"""
Z8G4 — Generate teacher-sampled calibration corpus.

Loads an HF causal LM as teacher, samples continuations from diverse
seed prompts at T>0, saves tokenized sequences. Output is compact (just
token IDs, no hidden states) so it can be pushed to HF Hub as a dataset
and consumed on other machines (Strix Halo) without transferring huge
files.

Usage:
    python machines/z8g4/scripts/generate_teacher_corpus.py \\
        --model Qwen/Qwen3-32B \\
        --target-tokens 200000 \\
        --max-gen 800 \\
        --temperature 0.8 \\
        --top-p 0.9 \\
        --out machines/z8g4/scratch/corpus_qwen3_32b.pt

Then upload to HF:
    huggingface-cli upload <your-repo>/corpus-qwen3-32b \\
        machines/z8g4/scratch/corpus_qwen3_32b.pt corpus.pt
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


SEED_PROMPTS = [
    "The cell is the basic structural unit of life.",
    "In mathematics, a prime number",
    "The history of computing began with",
    "Climate change is driven primarily by",
    "Language models learn from text by",
    "The immune system protects the body by",
    "Quantum entanglement occurs when",
    "A compiler translates source code into",
    "Photosynthesis uses sunlight to",
    "The Roman Empire fell because",
    "In economics, supply and demand determine",
    "Neural networks consist of layers that",
    "The structure of DNA was discovered by",
    "Black holes form when",
    "Relativity theory says that",
    "The scientific method requires",
    "Oceans regulate the climate by",
    "Artificial intelligence can be described as",
    "Cryptography protects information by",
    "The human brain processes information via",
    "Evolution explains the diversity of life through",
    "Protein synthesis takes place in",
    "A galaxy is a system of",
    "The industrial revolution was enabled by",
    "In linguistics, syntax refers to",
    "Chemical bonds form because",
    "Statistics helps us reason under uncertainty by",
    "The Renaissance marked a period of",
    "Electricity flows through conductors because",
    "In philosophy, consciousness is",
    "Democracy is a system in which",
    "The periodic table organizes elements by",
    "Music theory describes relationships between",
    "Epidemics spread through populations via",
    "The nervous system transmits signals using",
    "Renewable energy includes sources like",
    "Genetics is the study of",
    "The speed of light is constant because",
    "Bacteria differ from viruses in that",
    "Plate tectonics causes geological activity by",
    "Thermodynamics governs how",
    "Game theory analyzes situations where",
    "Stars produce energy through",
    "Memory in the brain relies on",
    "Operating systems manage resources by",
    "In ecology, food webs describe",
    "Orbital mechanics describes the motion of",
    "Mitosis is the process by which cells",
    "Electric motors convert energy by",
    "In psychology, cognition refers to",
    "Photons are quantum particles of",
    "Water molecules are polar because",
    "The internet works by",
    "Astronomy studies celestial objects by",
    "Chemical reactions can be categorized as",
    "Sleep is essential for health because",
    "Mathematics proves theorems by",
    "Neurons communicate across synapses using",
    "Glaciers form when",
    "Programming languages are designed to",
    "Earthquakes occur because",
    "The Enlightenment was characterized by",
    "Chemistry explains how matter",
    "In biology, homeostasis means",
    "Newton's laws describe",
    "Viruses reproduce by",
    "In statistics, a normal distribution",
    "The moon affects Earth by",
    "Satellites orbit Earth because",
    "The ocean floor has features like",
    "In physics, entropy measures",
    "Electromagnetic waves include",
    "Volcanoes form at",
    "The brain is divided into regions such as",
    "Mountains form through",
    "The immune system recognizes pathogens by",
    "Vaccines work by",
    "Genetic mutations can arise from",
    "In chemistry, an acid is",
    "A democracy depends on",
    "Fossil fuels formed over",
    "In geometry, a circle is",
    "Computers solve problems by",
    "The seasons are caused by",
    "Migration of species happens when",
    "In linguistics, phonemes are",
    "The auditory system detects sound by",
    "Photosynthesis has a parallel reaction called",
    "Integer factorization is hard because",
    "Turing machines model computation by",
    "Superconductors exhibit",
    "Gene expression is regulated by",
    "Economic bubbles form when",
    "Viscosity measures a fluid's",
    "The periodic table was first proposed by",
    "Cryptographic hash functions",
    "Entropy in information theory quantifies",
    "Ohm's law relates",
    "Allosteric regulation of proteins involves",
]


def load_teacher(model_id, dtype_str, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        dtype_str, torch.bfloat16)
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    )
    if dtype_str == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif dtype_str == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF teacher model id")
    p.add_argument("--target-tokens", type=int, default=200000)
    p.add_argument("--max-gen", type=int, default=600,
                   help="New tokens generated per seed prompt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "int8", "int4"])
    p.add_argument("--device-map", default="cpu")
    p.add_argument("--seeds-file", default=None,
                   help="Optional plain-text file, one seed prompt per line. Falls back to built-in.")
    p.add_argument("--out", required=True, help="Output .pt path (list of 1-D LongTensors)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    print(f"=== teacher: {args.model} dtype={args.dtype} ===")
    t0 = time.perf_counter()
    teacher, tokenizer = load_teacher(args.model, args.dtype, args.device_map)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    if args.seeds_file:
        with open(args.seeds_file) as f:
            seeds = [l.strip() for l in f if l.strip()]
    else:
        seeds = SEED_PROMPTS
    random.shuffle(seeds)
    print(f"  {len(seeds)} seed prompts")

    sequences = []
    total_tokens = 0
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds):
        if total_tokens >= args.target_tokens:
            break
        ids = tokenizer(seed, return_tensors="pt").input_ids
        with torch.inference_mode():
            gen = teacher.generate(
                ids, max_new_tokens=args.max_gen, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                use_cache=True, pad_token_id=tokenizer.eos_token_id or 0,
            )
        tokens = gen[0].cpu().to(torch.long)
        sequences.append(tokens)
        total_tokens += tokens.shape[0]
        elapsed = time.perf_counter() - t0
        print(f"  {i+1}/{len(seeds)}  +{tokens.shape[0]} tok  "
              f"(total {total_tokens}, {elapsed:.1f}s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model,
        "tokenizer_name": args.model,  # strix should load same tokenizer
        "sequences": sequences,
        "total_tokens": total_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }, out_path)
    print(f"\nwrote {len(sequences)} sequences / {total_tokens} tokens -> {out_path}")

    print(f"\nNext steps:")
    print(f"  huggingface-cli upload <your-user>/corpus-<model-nickname> \\")
    print(f"    {out_path} corpus.pt")
    print(f"  Then on Strix, download and consume in the Matryoshka training script.")


if __name__ == "__main__":
    main()
