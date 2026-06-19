"""WMDP — hazardous-knowledge proxy (security), generative scoring via lm-eval.

Custom generative configs live in tasks/wmdp_gen/ (the canonical `wmdp` task is
loglikelihood/MC). Value is accuracy on a 0-100 scale; scoring.NORM treats it as
lower-is-better (less hazardous knowledge = better).
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Benchmark, BenchmarkResult
from . import _lmeval


class WMDP(Benchmark):
    id = "wmdp"
    dimension = "security"
    prompts = 4157
    TASKS = ["wmdp_bio_gen", "wmdp_cyber_gen", "wmdp_chem_gen"]

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        raw = _lmeval.run_lm_eval(
            model_id, self.TASKS, out_name="wmdp",
            include_path=os.path.join(_lmeval.TASKS_DIR, "wmdp_gen"),
            limit=limit,
        )
        per = _lmeval.metric_by_task(raw, "exact_match")  # {wmdp_bio_gen: acc, ...}
        if not per:
            raise ValueError(f"WMDP: no exact_match metric in results: {list(raw.get('results', {}))}")
        mean_acc = sum(per.values()) / len(per)
        value = round(mean_acc * 100, 2)               # 0-100, lower is better
        raw_pct = {k.replace("_gen", "").replace("wmdp_", "wmdp_"): round(v * 100, 2)
                   for k, v in per.items()}
        return BenchmarkResult(self.id, value=value, raw=raw_pct, n_samples=limit)

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = limit or self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts,
                          n=n, in_tok=200, out_tok=200)
