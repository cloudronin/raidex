"""Shared helpers for the direct-call benchmarks (SimpleQA, StrongREJECT, XSTest).

Model responses and judge calls both go through litellm, so any provider works.
The judge model is read from config.yaml (judge_models.<id>), overridable with the
RAIDEX_JUDGE_MODEL env var. parallel_map runs per-item work concurrently (these
benchmarks make one model call + one judge call per item, so they must not be
serial over thousands of rows).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
import yaml

_CONFIG = None


def _config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        p = Path(__file__).resolve().parent.parent / "config.yaml"
        _CONFIG = yaml.safe_load(p.read_text())
    return _CONFIG


def judge_model(benchmark_id: str, default: str = "openai/gpt-4o") -> str:
    """Resolve the judge model for a benchmark. RAIDEX_JUDGE_MODEL overrides config."""
    env = os.environ.get("RAIDEX_JUDGE_MODEL")
    if env:
        return env
    jm = _config().get("judge_models", {}).get(benchmark_id, default)
    return jm if isinstance(jm, str) else default


def complete(model_id: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Single-turn litellm completion; returns the text (empty string on no content)."""
    resp = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def parallel_map(fn, items, workers: int | None = None):
    """Map fn over items concurrently, preserving order. Exceptions surface per item
    as the return of fn (callers should catch inside fn and return a sentinel).
    Worker count defaults to RAIDEX_CONCURRENCY (else 8) so throttling-prone
    providers (SambaNova, Mistral) can be run gently."""
    if workers is None:
        workers = int(os.environ.get("RAIDEX_CONCURRENCY", "8"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
