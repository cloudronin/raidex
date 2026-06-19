"""StrongREJECT — refusal robustness (security).

Loads the 313 forbidden prompts, gets the target model's responses via litellm,
and scores them with the StrongREJECT rubric evaluator (an LLM judge, provider-
agnostic via the models= kwarg). Reported value is the attack-success rate (mean
rubric score, 0-1); scoring.NORM treats it as lower-is-better. Calls run
concurrently. Small dataset — the runner runs it full.
"""
from __future__ import annotations

from typing import Optional

from .base import Benchmark, BenchmarkResult
from . import _direct


class StrongREJECT(Benchmark):
    id = "strongreject"
    dimension = "security"
    prompts = 313

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        from strong_reject.load_datasets import load_strongreject
        from strong_reject.evaluate import evaluate

        ds = load_strongreject()
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        # Default to the no-GPU rubric judge model; RAIDEX_JUDGE_MODEL overrides.
        jm = _direct.judge_model("strongreject", default="openai/gpt-4o-mini")

        def score_one(fp):
            # Per-pair evaluate() avoids evaluate_dataset's multiprocessing (dies on macOS).
            # API failures (model or judge, after retries) propagate to map_safe; a None
            # return = answered but the rubric produced no score (not a call failure).
            resp = _direct.complete(model_id, fp, max_tokens=512)
            out = evaluate(fp, resp, ["strongreject_rubric"], models=[jm])
            s = out[0].get("score")
            return float(s) if (s is not None and s == s) else None

        out, errors = _direct.map_safe(score_one, list(ds["forbidden_prompt"]), label="strongreject")
        scores = [s for s in _direct.oks(out) if s is not None]
        err = _direct.failure_error(len(errors), len(ds)) or (None if scores else "no valid rubric scores")
        asr = sum(scores) / len(scores) if scores else 0.0
        return BenchmarkResult(
            self.id, value=None if err else round(asr, 4),     # 0-1, lower is better (less harmful)
            raw={"asr": round(asr, 4), "n_scored": len(scores), "evaluator": "strongreject_rubric"},
            judge_model=jm.split("/")[-1], n_samples=len(scores),
            n_failed=len(errors), sample_errors=[e for _, e in errors[:3]], error=err,
        )

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = limit or self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts, n=n,
                          in_tok=60, out_tok=300, judge_calls=n, judge_in=600, judge_out=80,
                          judge_model=_direct.judge_model("strongreject", default="openai/gpt-4o-mini"))
