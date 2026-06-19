"""SimpleQA — factuality, direct litellm calls + judge, scored with the SimpleQA F1.

For each question the model answers, a judge classifies the answer as
CORRECT / INCORRECT / NOT_ATTEMPTED, and the headline metric is the SimpleQA F1
(harmonic mean of overall correctness and accuracy-given-attempted), 0-100.
Calls run concurrently via parallel_map.

NOTE: the grader below is a faithful adaptation of OpenAI simple-evals'
classification; swap in the verbatim GRADER_TEMPLATE for publication fidelity.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Optional

import pandas as pd

from .base import Benchmark, BenchmarkResult
from . import _direct

CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
CSV_PATH = "/tmp/raidex/simple_qa_test_set.csv"

GRADER = """You are grading whether a model's answer to a question is correct, given the gold answer.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

Classify the model answer as exactly one of:
- CORRECT: contains the gold answer or is semantically equivalent, with no contradicting facts.
- INCORRECT: contradicts the gold answer or asserts a wrong fact.
- NOT_ATTEMPTED: declines, says it doesn't know, or hedges without committing to an answer.

Reply with exactly one word: CORRECT, INCORRECT, or NOT_ATTEMPTED."""


class SimpleQA(Benchmark):
    id = "simpleqa"
    dimension = "factuality"
    prompts = 4326

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        df = _load_csv()
        if limit:
            df = df.head(limit)
        jm = _direct.judge_model("simpleqa")

        def grade(rec):
            # Let API failures (after retries) propagate — map_safe counts them. Only a
            # genuine answer the judge can't place falls through to "C" (not_attempted).
            q, gold = rec["problem"], rec["answer"]
            answer = _direct.complete(model_id, q, max_tokens=512)
            verdict = _direct.complete(
                jm, GRADER.format(question=q, gold=gold, answer=answer), max_tokens=8
            ).upper()
            if "NOT_ATTEMPTED" in verdict:
                return "C"
            if "INCORRECT" in verdict:
                return "B"
            if "CORRECT" in verdict:
                return "A"
            return "C"

        out, errors = _direct.map_safe(grade, df.to_dict("records"), label="simpleqa")
        grades = _direct.oks(out)
        err = _direct.failure_error(len(errors), len(df))
        n = len(grades)
        n_correct = grades.count("A")
        n_incorrect = grades.count("B")
        n_not_attempted = grades.count("C")
        is_correct = n_correct / n if n else 0.0
        acc_given = n_correct / (n_correct + n_incorrect) if (n_correct + n_incorrect) else 0.0
        f1 = (2 * acc_given * is_correct / (acc_given + is_correct)) if (acc_given + is_correct) else 0.0
        return BenchmarkResult(
            self.id, value=None if err else round(f1 * 100, 2),     # 0-100, higher is better
            raw={"correct": n_correct, "incorrect": n_incorrect, "not_attempted": n_not_attempted,
                 "is_correct": round(is_correct, 4), "accuracy_given_attempted": round(acc_given, 4),
                 "f1": round(f1, 4)},
            judge_model=jm.split("/")[-1], n_samples=n,
            n_failed=len(errors), sample_errors=[e for _, e in errors[:3]], error=err,
        )

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = limit or self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts, n=n,
                          in_tok=60, out_tok=120, judge_calls=n, judge_in=200, judge_out=4,
                          judge_model=_direct.judge_model("simpleqa"))


def _load_csv() -> "pd.DataFrame":
    os.makedirs("/tmp/raidex", exist_ok=True)
    if not os.path.exists(CSV_PATH):
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    return pd.read_csv(CSV_PATH)
