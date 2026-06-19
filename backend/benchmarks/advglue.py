"""AdvGLUE — robustness (Tier B), adversarial-perturbation GLUE classification.

Robust accuracy via exact-match label classification (no judge, no GPU). Data:
AI-Secure/adv_glue (ungated parquet; AdvGLUE base, human-validated adversarial GLUE).
Value is overall accuracy on adversarial inputs, 0-1, higher is better. Calls run
concurrently. The dataset is small (~738), so the runner runs it full.
"""
from __future__ import annotations

import re
from typing import Optional

from datasets import load_dataset

from .base import Benchmark, BenchmarkResult
from . import _direct


def _sst2(r):
    return (f"Classify the sentiment of this movie-review sentence.\nSentence: {r['sentence']}\n"
            "Answer with one word: positive or negative.")


def _mnli(r):
    return (f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}\n"
            "Does the premise entail the hypothesis, contradict it, or neither? "
            "Answer with one word: entailment, contradiction, or neutral.")


def _qnli(r):
    return (f"Question: {r['question']}\nSentence: {r['sentence']}\n"
            "Does the sentence contain the answer to the question? Answer yes or no.")


def _rte(r):
    return (f"Text 1: {r['sentence1']}\nText 2: {r['sentence2']}\n"
            "Does Text 1 entail Text 2? Answer yes or no.")


def _qqp(r):
    return (f"Question 1: {r['question1']}\nQuestion 2: {r['question2']}\n"
            "Do these two questions have the same meaning? Answer yes or no.")


# config -> (prompt_fn, {gold_label_index: [match keywords]})
TASKS = {
    "adv_sst2":            (_sst2, {0: ["negative"], 1: ["positive"]}),
    "adv_mnli":            (_mnli, {0: ["entail"], 1: ["neutral"], 2: ["contradict"]}),
    "adv_mnli_mismatched": (_mnli, {0: ["entail"], 1: ["neutral"], 2: ["contradict"]}),
    "adv_qnli":            (_qnli, {0: ["yes"], 1: ["no"]}),   # 0 = answer present (entailment)
    "adv_rte":             (_rte,  {0: ["yes"], 1: ["no"]}),   # 0 = entailment
    "adv_qqp":             (_qqp,  {0: ["no"], 1: ["yes"]}),   # 0 = not duplicate
}


def _classify(resp: str, label_map: dict):
    r = resp.lower()
    matched = set()
    for idx, kws in label_map.items():
        for kw in kws:
            hit = re.search(rf"\b{re.escape(kw)}\b", r) if len(kw) <= 3 else (kw in r)
            if hit:
                matched.add(idx)
                break
    return next(iter(matched)) if len(matched) == 1 else None


class AdvGLUE(Benchmark):
    id = "advglue"
    dimension = "robustness"
    prompts = 738

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        items = []
        for cfg, (pf, lm) in TASKS.items():
            rows = list(load_dataset("AI-Secure/adv_glue", cfg)["validation"])
            if limit:
                rows = rows[:limit]
            for row in rows:
                items.append((cfg, pf, lm, row))

        def judge(it):
            cfg, pf, lm, row = it
            try:
                pred = _classify(_direct.complete(model_id, pf(row), max_tokens=16), lm)
            except Exception:
                pred = None
            return (cfg, 1 if (pred is not None and pred == int(row["label"])) else 0)

        res = _direct.parallel_map(judge, items)
        total = len(res)
        correct = sum(ok for _, ok in res)
        by_task: dict[str, list[int]] = {}
        for cfg, ok in res:
            d = by_task.setdefault(cfg, [0, 0])
            d[0] += ok
            d[1] += 1
        raw = {cfg.replace("adv_", ""): round(c / n, 4) for cfg, (c, n) in by_task.items()}
        return BenchmarkResult(self.id, value=round(correct / total, 4) if total else 0.0,
                               raw=raw, n_samples=total)

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = (limit * len(TASKS)) if limit else self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts, n=n, in_tok=90, out_tok=8)
