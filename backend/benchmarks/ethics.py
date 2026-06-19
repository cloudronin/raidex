"""ETHICS — machine ethics, generative scoring via lm-eval.

Per METHODOLOGY, the ETHICS constituent is the mean accuracy over four moral-
judgment subtasks: justice, deontology, virtue, utilitarianism. Custom generative
configs live in tasks/ethics_gen/ (canonical tasks are loglikelihood/MC). Value is
0-1, higher is better.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Benchmark, BenchmarkResult
from . import _lmeval


class ETHICS(Benchmark):
    id = "ethics"
    dimension = "machine_ethics"
    prompts = 13000
    TASKS = [
        "ethics_justice_gen", "ethics_deontology_gen",
        "ethics_virtue_gen", "ethics_utilitarianism_gen",
    ]

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        raw = _lmeval.run_lm_eval(
            model_id, self.TASKS, out_name="ethics",
            include_path=os.path.join(_lmeval.TASKS_DIR, "ethics_gen"),
            limit=limit,
        )
        per = _lmeval.metric_by_task(raw, "exact_match")
        if not per:
            raise ValueError(f"ETHICS: no exact_match metric in results: {list(raw.get('results', {}))}")
        value = round(sum(per.values()) / len(per), 4)        # 0-1, higher is better
        raw_acc = {k.replace("ethics_", "").replace("_gen", ""): round(v, 4)
                   for k, v in per.items()}
        return BenchmarkResult(self.id, value=value, raw=raw_acc, n_samples=limit)

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = limit or self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts,
                          n=n, in_tok=120, out_tok=8)
