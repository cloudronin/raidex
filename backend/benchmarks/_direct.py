"""Shared helpers for the direct-call benchmarks (SimpleQA, StrongREJECT, XSTest,
AdvGLUE, ConfAIde).

Model and judge calls both go through litellm, so any provider works. ``complete``
retries with backoff (litellm ``num_retries``) so transient rate-limits (429) and
timeouts don't surface as failures; only a call that still fails AFTER the retries
raises. ``map_safe`` runs per-item work concurrently and — crucially — does NOT
swallow those post-retry failures into degenerate data: it returns them so the
wrapper can guard (report the benchmark as errored, excluded from coverage) and the
runner can DLQ them for targeted replay. This is the fix for the silent-degeneration
bug where rate-limited calls were scored as wrong and folded into a fake 8/8.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import yaml

# Drop provider-unsupported params instead of raising — mirrors the lm-eval proxy's
# --drop_params. Critical for models like claude-opus-4-8 that reject temperature=0
# (only temperature=1 allowed); without this, every direct call to them errors out.
litellm.drop_params = True

# Models whose API rejects temperature=0 (only the default value, 1, is allowed) — e.g.
# GPT-5.5, which shipped reasoning-locked. drop_params can't help here: temperature is a
# *supported* param and only the *value* 0 is rejected, but drop_params strips unsupported
# param *names*, not values. So we omit temperature for these models instead. This is a
# known seed; complete() also discovers new ones reactively, and the lm-eval path reads
# this set to force temperature=1. (gpt-5.2 and earlier accept temperature=0 fine — this
# is 5.5-specific, hence a known-set + reactive catch, not a "gpt-5*" name heuristic.)
REJECTS_TEMP0 = {"openai/gpt-5.5"}
_NO_TEMP0 = set(REJECTS_TEMP0)

_CONFIG = None

# Retry/backoff for transient provider errors (429 / timeout). litellm performs the
# backoff internally; a call only raises once it still fails after this many retries.
NUM_RETRIES = int(os.environ.get("RAIDEX_NUM_RETRIES", "6"))
TIMEOUT = float(os.environ.get("RAIDEX_TIMEOUT", "120"))
# Global litellm timeout so EVERY in-process litellm call fails fast instead of hanging
# on a stalled connection — including library calls we don't wrap (e.g. strong_reject's
# own rubric-judge call, which has no timeout and dead-locked a run for 15 min).
litellm.request_timeout = TIMEOUT
# If more than this fraction of a benchmark's calls fail after retries, the value is
# unreliable: the wrapper reports an error (excluded from coverage) rather than fold
# degenerate scores into the composite.
MAX_FAILURE_RATE = float(os.environ.get("RAIDEX_MAX_FAILURE_RATE", "0.25"))
# Emit a per-request progress counter ("<bench>: done/total") every N completions, so a
# slow/hung benchmark is visible instead of silent. Set RAIDEX_PROGRESS_EVERY=0 to mute.
PROGRESS_EVERY = int(os.environ.get("RAIDEX_PROGRESS_EVERY", "25"))


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
    """Single-turn litellm completion with retry/backoff.

    Returns the response text (may be empty if the model genuinely produced no
    content). Raises only if the call still fails after ``NUM_RETRIES`` — callers
    must let that propagate (route it through ``map_safe``), so failures are counted
    and DLQ'd rather than silently scored as wrong.
    """
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        num_retries=NUM_RETRIES,
        timeout=TIMEOUT,
    )
    if model_id not in _NO_TEMP0:
        kwargs["temperature"] = temperature
    try:
        resp = litellm.completion(**kwargs)
    except litellm.BadRequestError as e:
        # A model that rejects temperature=0 (only the default, 1, allowed — e.g. gpt-5.5).
        # drop_params can't strip a disallowed *value*, so retry without temperature and
        # remember the model so later calls skip it directly (no repeated double-call).
        if "temperature" in kwargs and "temperature" in str(e).lower():
            _NO_TEMP0.add(model_id)
            kwargs.pop("temperature", None)
            resp = litellm.completion(**kwargs)
        else:
            raise
    return (resp.choices[0].message.content or "").strip()


def map_safe(fn, items, workers: int | None = None, label: str | None = None):
    """Map ``fn`` over ``items`` concurrently, order-preserving.

    ``fn`` should RAISE on a failed API call (after ``complete``'s retries) — do not
    catch it inside ``fn``. Returns ``(out, errors)``:
      * ``out``    — list aligned with ``items`` of ``(ok: bool, result)``; ``result``
                     is ``None`` when ``ok`` is False.
      * ``errors`` — list of ``(index, "ErrType: msg")`` for items whose ``fn`` raised.

    A ``None`` *returned* by ``fn`` (vs raised) is a valid "answered but unparseable"
    result and is NOT counted as a failure. Worker count defaults to
    RAIDEX_CONCURRENCY so throttling-prone providers can be run gently. If ``label`` is
    set, prints a live "label: done/total (n failed)" counter as requests complete.
    """
    if workers is None:
        workers = int(os.environ.get("RAIDEX_CONCURRENCY", "8"))
    items = list(items)
    total = len(items)

    def _wrap(ix, item):
        try:
            return (ix, True, fn(item), None)
        except Exception as e:  # post-retry failure — surfaced, never swallowed
            return (ix, False, None, f"{type(e).__name__}: {str(e)[:200]}")

    results: list = [None] * total
    oks_: list = [False] * total
    errors: list = []
    done = 0
    # submit + as_completed (not ex.map) so we can count each request as it lands.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_wrap, ix, it) for ix, it in enumerate(items)]
        for fut in as_completed(futs):
            ix, ok, res, err = fut.result()
            results[ix] = res
            oks_[ix] = ok
            if not ok:
                errors.append((ix, err))
            done += 1
            if label and PROGRESS_EVERY and total and (done % PROGRESS_EVERY == 0 or done == total):
                nf = len(errors)
                print(f"  · {label}: {done}/{total}" + (f"  ({nf} failed)" if nf else ""), flush=True)
    out = [(oks_[i], results[i]) for i in range(total)]
    return out, errors


def oks(out):
    """Successful results from a ``map_safe`` ``out`` list (drops failed items)."""
    return [res for ok, res in out if ok]


def failure_error(n_failed: int, n_total: int) -> str | None:
    """Error string if too many calls failed after retries (or none ran), else None."""
    if n_total == 0:
        return "no samples evaluated (0 calls)"
    if n_failed / n_total > MAX_FAILURE_RATE:
        return (f"{n_failed}/{n_total} API calls failed after {NUM_RETRIES} retries "
                f"(rate-limit/timeout/unsupported-param — see DLQ sample_errors); result unreliable")
    return None


def parallel_map(fn, items, workers: int | None = None):
    """Plain concurrent map (no failure tracking). Retained for callers that handle
    their own error semantics; new code should prefer ``map_safe``."""
    if workers is None:
        workers = int(os.environ.get("RAIDEX_CONCURRENCY", "8"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
