"""API-cost estimation for ``--dry-run``.

Uses litellm's pricing map when the model is known; otherwise falls back to the
spec's per-benchmark order-of-magnitude full-run estimates, scaled by the sample
fraction. Never makes API calls — pricing lookup only.
"""
from __future__ import annotations

# Spec Tier A per-full-run estimates (raidex-spec-v3_5.md Benchmarks table).
_FALLBACK_FULL_USD = {
    "bbq": 10.0, "wmdp": 8.0, "simpleqa": 30.0,
    "strongreject": 5.0, "ethics": 5.0, "xstest": 4.0,
    "advglue": 2.0, "confaide": 1.0,
}


def token_cost(model_id, *, benchmark_id, full_n, n, in_tok, out_tok,
               judge_calls=0, judge_in=0, judge_out=0, judge_model="openai/gpt-4o"):
    """Estimate USD for ``n`` prompts of a benchmark, plus optional judge calls.

    Falls back to ``_FALLBACK_FULL_USD[benchmark_id] * (n / full_n)`` when litellm
    has no pricing for ``model_id`` (e.g. OpenAI-compatible endpoints like GLM-5).
    """
    try:
        from litellm import cost_per_token
        pin, pout = cost_per_token(model=model_id, prompt_tokens=in_tok, completion_tokens=out_tok)
        total = (pin + pout) * n
        if judge_calls:
            jin, jout = cost_per_token(model=judge_model, prompt_tokens=judge_in, completion_tokens=judge_out)
            total += (jin + jout) * judge_calls
        return round(total, 2)
    except Exception:
        full = _FALLBACK_FULL_USD.get(benchmark_id, 10.0)
        return round(full * (n / full_n), 2)
