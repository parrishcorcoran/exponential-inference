# exponential-inference

**Status: in construction — Stage 0 scaffold.**

This repository demonstrates a measurement result on Microsoft's BitNet b1.58 2B:
inference accelerates as generation proceeds, because the model is at a
spin-glass ground state and its per-token manifold dimensionality collapses
predictably with context length.

This is *not* a compression technique. The rank requirements we exploit are
extracted from the model's own measured geometry at inference time.

A full writeup will replace this placeholder after Stage 4 completes. See
`STAGES.md` for the current construction plan and checkpoints.
