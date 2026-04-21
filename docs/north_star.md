# North Star — the smartest, smallest model in the world

*A short vision document. Read the main README for the project background.*

---

## The goal

Not a speculative-decoding drafter. Not a distilled compression of a big
teacher. **A tiny model (~30-50M params) that matches or exceeds any
conventionally-trained teacher at any size**, because it's trained
against a measured manifold rather than against a bigger model's
approximation of one.

## The architectural recipe (current best guess)

Two artifacts, shipped together:

- **manifold.pt** — a precomputed map of the tokenizer family's hidden-
  state geometry. Includes per-layer PCA bases, rotation operators,
  carry/flip subspaces, stabilization reference points. ~50 MB per
  tokenizer family. Shared across all downstream models.
- **student.pt** — the tiny model. Trained to traverse the manifold
  given context. Because the manifold map handles "where tokens live,"
  the student's weights only encode "how to choose the next move."
  Estimated 30-50M params at useful quality.

Both are loaded at runtime. Inference = read current context, consult
manifold map for position candidates, use student weights to choose
traversal, emit next token.

## Why this is a different paradigm

Conventional LLM: all the knowledge is entangled in the weights. Bigger
weights = more memorized patterns + more room for the approximation to
be close to the truth. Small models fail because the approximation is
coarse and they have no way to access what they lost.

This project: the knowledge is split. The **manifold map** carries the
geometric truth (what every token could reasonably be in every context,
at every layer depth). The **student** carries the traversal policy
(what to do with that knowledge given the current sequence).

The student doesn't need to memorize the whole world because the world's
geometry is already in the map. It just needs to know how to walk on
the map competently. That's a much smaller problem than building a
teacher that approximates everything the manifold contains.

## What makes this possible

- **Finding 01**: manifold dim ~9-11, universal per tokenizer family.
  Small enough to represent completely.
- **Finding 11**: the forward pass is an RG flow + quantum-measurement-
  like purification. Geometric traversal, not symbolic retrieval. A
  small model respecting geometry is in principle sufficient.
- **Finding 10 (corrected)**: boundary compressible, bulk preserved. The
  architectural discipline that keeps a small model "complete" rather
  than lossy.
- **Stages 58/59 (candidate Finding 14)**: the manifold has
  locally-bimodal rotation structure + walking-basis drift. A compact
  description a small model can learn.

## What would close the claim

Three measurements, ordered by cost:

1. **Multi-teacher ensemble manifold map**. Z8G4's current priority.
   Average embedding-geometry PCA bases across Qwen3 family. Upload
   as `manifold.pt` artifact.
2. **Student perplexity < teacher perplexity on held-out**. Measure
   `(student_ppl − teacher_ppl)` during scaled training; crossover
   point is the ceiling-break. (Strix Halo.)
3. **Deployment demo**. Tiny model + manifold map on phone or browser,
   matching a 14B cloud model's output quality. Visible proof.

## What this changes for all prior work

- **Holographic Matryoshka (Finding 10)** is no longer "the technique."
  It's the architectural substrate. Its boundary/bulk distinction tells
  us what the small model has to respect.
- **Speculative decoding** is a side application, not the goal. Stage
  54d's confidence-stratified agreement is a consequence of the manifold-
  target training, not its purpose.
- **Matryoshka weight factoring** is dead for quality (Finding 10
  correction). The manifold-target training replaces it.
- **The RG + quantum measurement framing (Finding 11)** becomes the
  theoretical justification. The forward pass is geometry; a small
  model respecting the geometry is enough.

## Why this is worth pursuing

Current LLM deployment economics assume "bigger model = better quality =
more cloud compute." If we can ship a ~100 MB artifact + tiny local
compute that matches a 14B cloud model, the economics collapse.

The deeper claim: language is geometric. Every LLM is approximating the
same underlying geometry. We can measure it directly and train against
the measurement. The result beats any teacher approximation because
it's closer to the truth.

That's the north star. Everything else — the specific architecture,
training recipe, compression technique — is engineering in service of
this goal.
