"""BBQ — fairness & bias, via lm-eval-harness (generative scoring).

BBQ is natively a multiple-choice/loglikelihood task; frontier chat APIs don't
expose logprobs, so we use lm-eval's shipped ``bbq_generate`` (output_type:
generate_until), which prompts for an answer and extracts the chosen option from
text. See METHODOLOGY "Scoring method".
"""
from __future__ import annotations

from typing import Optional

from .base import Benchmark, BenchmarkResult
from . import _lmeval


class BBQ(Benchmark):
    id = "bbq"
    dimension = "fairness_bias"
    prompts = 8800
    TASK = "bbq_generate"

    def run(self, model_id: str, limit: Optional[int] = None) -> BenchmarkResult:
        # bbq_generate has no generation_kwargs, so lm-eval's default until=["\n\n"]
        # (whitespace) is sent — Anthropic rejects that. Override with a
        # non-whitespace sentinel + token cap via gen_kwargs.
        raw = _lmeval.run_lm_eval(
            model_id, self.TASK, out_name="bbq",
            gen_kwargs="until=STOPSEQ,max_gen_toks=256", limit=limit,
        )
        accs = _lmeval.metric_by_task(raw, "acc")  # {"bbq_generate": acc}
        acc = accs.get(self.TASK) or (sum(accs.values()) / len(accs) if accs else None)
        return BenchmarkResult(
            benchmark_id=self.id,
            value=acc,                       # 0-1, higher is better
            raw=raw.get("results", raw),
            n_samples=limit,
        )

    def estimate_cost(self, model_id: str, limit: Optional[int] = None) -> float:
        from cost import token_cost
        n = limit or self.prompts
        return token_cost(model_id, benchmark_id=self.id, full_n=self.prompts,
                          n=n, in_tok=350, out_tok=48)
