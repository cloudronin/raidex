"""Shared lm-eval-harness runner for the generative MC benchmarks (BBQ/WMDP/ETHICS).

Starts a litellm proxy (--drop_params), runs lm_eval against it via the
chat-completions backend, and returns parsed results. Generative scoring + answer
extraction is configured per task (the shipped ``bbq_generate``, or our custom
configs under ``tasks/``). See METHODOLOGY "Scoring method".
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Optional

from ._proxy import litellm_proxy

OUT_ROOT = os.environ.get("RAIDEX_OUT_ROOT", "/tmp/raidex")
# Repo-root /tasks dir holding our custom generative task configs.
TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")


def run_lm_eval(model_id: str, tasks, *, out_name: str,
                gen_kwargs: Optional[str] = None, include_path: Optional[str] = None,
                limit: Optional[int] = None, num_concurrent: Optional[int] = None) -> dict:
    """Run lm_eval for ``tasks`` against ``model_id`` through a litellm proxy.

    ``gen_kwargs`` (CLI string) overrides a task's generation_kwargs — needed for
    the shipped ``bbq_generate`` (whose default whitespace ``until`` Anthropic
    rejects). Our custom configs set their own Anthropic-safe ``until: []``, so
    they pass ``gen_kwargs=None``. Returns the parsed results.json dict.
    """
    if num_concurrent is None:
        num_concurrent = int(os.environ.get("RAIDEX_LMEVAL_CONCURRENCY",
                                            os.environ.get("RAIDEX_CONCURRENCY", "8")))
    max_retries = int(os.environ.get("RAIDEX_LMEVAL_RETRIES", "6"))
    out_dir = os.path.join(OUT_ROOT, out_name)
    os.makedirs(out_dir, exist_ok=True)
    task_arg = ",".join(tasks) if isinstance(tasks, (list, tuple)) else tasks
    with litellm_proxy(model_id) as (base_url, served):
        cmd = [
            "lm_eval", "--model", "local-chat-completions",
            "--model_args",
            f"model={served},base_url={base_url}/chat/completions,"
            f"num_concurrent={num_concurrent},max_retries={max_retries}",
            "--apply_chat_template",
            "--tasks", task_arg,
            "--output_path", out_dir,
        ]
        if gen_kwargs:
            cmd += ["--gen_kwargs", gen_kwargs]
        if include_path:
            cmd += ["--include_path", include_path]
        if limit:
            cmd += ["--limit", str(limit)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # Surface the real lm_eval error (rate-limit, proxy death, bad param) so the
            # runner can DLQ a meaningful reason instead of an opaque CalledProcessError.
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError(
                f"lm_eval exited {proc.returncode} for tasks={task_arg} model={model_id}: ...{tail}"
            )
    return _load_results(out_dir)


def _load_results(out_dir: str) -> dict:
    candidates = sorted(
        glob.glob(os.path.join(out_dir, "**", "results*.json"), recursive=True)
        + glob.glob(os.path.join(out_dir, "results*.json")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No lm-eval results JSON under {out_dir}")
    with open(candidates[-1]) as f:
        return json.load(f)


def metric_by_task(raw: dict, metric_name: str) -> dict:
    """Map each results task -> the value of the metric whose name (before the
    comma-separated filter key) equals ``metric_name``. E.g. ``'acc'`` matches
    ``'acc,none'``; ``'exact_match'`` matches ``'exact_match,flexible-extract'``."""
    out: dict[str, float] = {}
    res = raw.get("results", raw)
    for task, metrics in res.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k.split(",")[0] == metric_name:
                out[task] = float(v)
                break
    return out
