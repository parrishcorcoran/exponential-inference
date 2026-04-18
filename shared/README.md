# Shared coordination

Multiple machines contribute to this project. This doc is the sync
contract so nobody overwrites anybody.

## Principle

Each machine owns its own folder: `machines/<name>/`. Never write to
another machine's folder from a different machine.

Shared artifacts come in two tiers:

1. **Small (committed to git):** JSON results, markdown summaries,
   small plots. Pulled via `git pull` from anywhere.
2. **Large (on HuggingFace Hub):** corpora, model weights, activation
   caches. Pulled via `huggingface-cli download`.

Nothing heavy in git. Nothing transient / machine-specific in shared.

## Machines

| machine | folder | role | size |
|---|---|---|---|
| Z8G4 (HP workstation, 700 GB RAM, no GPU, Skylake Xeon) | `machines/z8g4/` | Big-model measurement, teacher-corpus generation, overnight eval | 70B+ OK |
| Strix Halo (82 GB VRAM, ROCm) | `machines/strix_halo/` | Interactive distillation training, ROCm inference benchmarks | up to 32B |
| MacBook (MPS, base repo) | `scripts/`, `src/`, `results/` | Prototype / small-model experiments only | up to 4B |

## What goes where

### Committed to git
- `machines/<name>/results/*.json` — small experiment outputs per machine.
- `machines/<name>/scripts/` — scripts unique to that machine.
- `machines/<name>/README.md` — role + setup.
- `docs/research_context.md` — shared research framing (single file, one machine edits at a time — coordinate via commits).
- Top-level `scripts/` and `results/` — the shared stage 1-15 prototyping on MacBook.

### NOT committed to git (gitignored)
- `machines/<name>/scratch/` — local scratch per machine.
- Any `.safetensors`, `.bin`, `.pt`, large `.npy`.

### Pushed to HuggingFace Hub
- Corpora from Z8G4: `<user>/exponential-inference-corpus-<model>`.
- Student weights from Strix: `<user>/exponential-inference-student-<tag>`.
- Measurement caches (hidden states) if ever needed cross-machine.

### Cross-machine read/write rules
- A machine may READ other machines' committed git folders (e.g. Strix reads Z8G4's published manifold JSONs).
- A machine may NEVER WRITE to another machine's folder. Own your own folder.
- `docs/` and `shared/` are coordination zones — edit with judgment, announce big changes in commit messages.

## Typical end-to-end flow

1. **Z8G4**: pick a big teacher. Run `measure_manifold_large.py`. Commit JSON to `machines/z8g4/results/`. Push.
2. **Z8G4**: run `generate_teacher_corpus.py` for the same teacher. Upload corpus.pt to HF Hub.
3. **Strix Halo**: pull git, download corpus from HF, run `train_matryoshka.py` with local teacher. Upload student to HF.
4. **Z8G4** (later): pull student from HF, run `validate_student.py` on large held-out text set. Commit summary JSON to `machines/z8g4/results/`.

## Naming conventions

- Corpus files: `corpus-<model-nickname>.pt`
- Student artifacts: `student-<teacher-nickname>-r<rank>.pt`
- Results: `<stage-or-experiment>_<model>.json`

Keep nicknames short and consistent. E.g. `qwen3-32b`, `llama3-70b`.
