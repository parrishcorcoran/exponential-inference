"""Per-position wall-clock measurement for the acceleration curve.

The measurement is intentionally simple: for each prompt we generate
``max_new_tokens`` tokens one at a time, recording the wall-clock
duration of each ``model.forward`` call. We do this for two modes:

  - ``base``: the unwrapped model.
  - ``dynamic``: the model with ``DynamicRankBitNet`` hooks installed.

We use ``use_cache=True`` so the per-step forward only processes the
new token (length 1) plus KV cache, exactly as real deployments would.

CUDA/HIP timings use ``torch.cuda.synchronize`` around each forward;
CPU timings use ``time.perf_counter`` only. GPU warmup runs one forward
before recording.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from src.inference.dynamic_rank import DynamicRankBitNet, RankStats


@dataclass
class PerPromptRun:
    prompt_id: str
    prompt_text: str
    prompt_token_count: int
    mode: str  # "base" or "dynamic"
    per_step_seconds: List[float] = field(default_factory=list)
    mean_rank_per_position: List[float] = field(default_factory=list)  # len = max_new_tokens
    generated_text: str = ""
    generated_token_ids: List[int] = field(default_factory=list)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _greedy_step(model, input_ids: torch.Tensor, past_kv):
    """One greedy generation step with KV cache. Returns
    ``(next_token_id, new_past_kv)``.
    """
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
    logits = out.logits[:, -1, :]
    tok = logits.argmax(dim=-1, keepdim=True)
    return tok, out.past_key_values


def generate_and_time(
    model,
    tokenizer,
    prompt_text: str,
    prompt_id: str,
    max_new_tokens: int,
    device: torch.device,
    wrapper_factory: Optional[Callable[[], DynamicRankBitNet]] = None,
    mode_label: str = "base",
    warmup_steps: int = 2,
) -> PerPromptRun:
    """Generate up to ``max_new_tokens`` tokens and record per-step timings.

    If ``wrapper_factory`` is provided, the returned DynamicRankBitNet
    is installed for the duration of the generation and its per-step
    mean rank (averaged across target layers) is recorded alongside
    per-step wall-clock.
    """
    model.eval()
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    run = PerPromptRun(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        prompt_token_count=prompt_len,
        mode=mode_label,
    )

    ctx = wrapper_factory() if wrapper_factory is not None else None
    wrapper: Optional[DynamicRankBitNet] = None
    if ctx is not None:
        wrapper = ctx.install()

    try:
        # Prefill is not counted — it's a one-shot cost that the
        # per-position decoding curve should not be polluted by.
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
        past_kv = out.past_key_values
        last_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Warmup: run a few steps untimed so autograd / allocator / hip
        # caches are hot.
        _sync(device)
        for _ in range(warmup_steps):
            last_tok, past_kv = _greedy_step(model, last_tok, past_kv)
        _sync(device)
        # Reset telemetry after warmup so only real steps are recorded.
        if wrapper is not None:
            wrapper.stats = RankStats()

        # Record step-by-step.
        for step in range(max_new_tokens):
            _sync(device)
            t0 = time.perf_counter()
            last_tok, past_kv = _greedy_step(model, last_tok, past_kv)
            _sync(device)
            dt = time.perf_counter() - t0
            run.per_step_seconds.append(dt)
            run.generated_token_ids.append(int(last_tok[0, 0].item()))

            if wrapper is not None:
                # Take the latest per-layer rank scalar recorded during this step.
                layer_ranks = []
                for li, lst in wrapper.stats.ranks_per_layer.items():
                    if lst:
                        layer_ranks.append(float(lst[-1].float().mean().item()))
                run.mean_rank_per_position.append(
                    sum(layer_ranks) / max(1, len(layer_ranks))
                )
            else:
                run.mean_rank_per_position.append(float("nan"))

        run.generated_text = tokenizer.decode(run.generated_token_ids,
                                              skip_special_tokens=True)
    finally:
        if wrapper is not None:
            wrapper.remove()

    return run


def aggregate_curve(runs: List[PerPromptRun]) -> Dict[str, List[float]]:
    """Average per-step timings across prompts, returning mean and sem.

    All runs must have the same ``max_new_tokens``. Prompts with
    shorter runs are padded with NaN and the averages ignore them.
    """
    if not runs:
        return {"mean_seconds": [], "sem_seconds": [], "n": []}
    import math
    max_len = max(len(r.per_step_seconds) for r in runs)
    means: List[float] = []
    sems: List[float] = []
    counts: List[int] = []
    for pos in range(max_len):
        vals = [r.per_step_seconds[pos] for r in runs if pos < len(r.per_step_seconds)]
        if not vals:
            means.append(float("nan"))
            sems.append(float("nan"))
            counts.append(0)
            continue
        m = sum(vals) / len(vals)
        if len(vals) > 1:
            v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
            sem = math.sqrt(v / len(vals))
        else:
            sem = 0.0
        means.append(m)
        sems.append(sem)
        counts.append(len(vals))
    return {"mean_seconds": means, "sem_seconds": sems, "n": counts}


def speedup_curve(base: Dict, dynamic: Dict) -> Dict[str, List[float]]:
    """Pointwise ratio base/dynamic of mean per-step times.

    Returned lengths match the shorter of the two input curves.
    """
    n = min(len(base["mean_seconds"]), len(dynamic["mean_seconds"]))
    ratio = []
    for i in range(n):
        b = base["mean_seconds"][i]
        d = dynamic["mean_seconds"][i]
        if d and d > 0:
            ratio.append(b / d)
        else:
            ratio.append(float("nan"))
    return {"ratio": ratio, "n": n}
