"""Calibration: generative (regex-extraction) vs canonical loglikelihood MCQ scoring.

Raidex scores BBQ/WMDP/ETHICS *generatively* (chat APIs lack logprobs). The worry:
does generative answer-extraction distort the score vs the canonical loglikelihood
method the published benchmarks use? This runs lm-eval's `hf` backend (real logprobs)
on a small open-weight model and scores each benchmark BOTH ways on the SAME items:
  - loglikelihood: the native lm-eval MCQ task (no chat template) = canonical
  - generative:    our generate_until configs + regex extraction (chat template) = Raidex's method
A small per-benchmark delta + same ordering => generative scoring is a faithful proxy,
ruling it out as the cause of the leaderboard finding.

Env: CAL_MODEL (default Qwen2.5-1.5B-Instruct), CAL_LIMIT (100), CAL_DEVICE (mps),
CAL_BATCH (8). Same first-N items per task in both modes (lm-eval --limit is deterministic).
"""
import glob
import json
import os
import subprocess

MODEL = os.environ.get("CAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LIMIT = os.environ.get("CAL_LIMIT", "100")
DEVICE = os.environ.get("CAL_DEVICE", "mps")
BATCH = os.environ.get("CAL_BATCH", "8")
TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
OUT = "/tmp/raidex_cal"

# label, loglikelihood task(s), generative task(s), include_path, gen_kwargs, gen-metric, ll-metric
SUITES = [
    ("bbq", "bbq", "bbq_generate", None, "until=STOPSEQ,max_gen_toks=256", "acc", "acc"),
    ("wmdp", "wmdp_bio,wmdp_cyber,wmdp_chem",
     "wmdp_bio_gen,wmdp_cyber_gen,wmdp_chem_gen",
     os.path.join(TASKS_DIR, "wmdp_gen"), None, "exact_match", "acc"),
    ("ethics", "ethics_deontology,ethics_justice,ethics_utilitarianism,ethics_virtue",
     "ethics_deontology_gen,ethics_justice_gen,ethics_utilitarianism_gen,ethics_virtue_gen",
     os.path.join(TASKS_DIR, "ethics_gen"), None, "exact_match", "acc"),
]


def run(tasks, include_path, gen_kwargs, chat, tag):
    od = os.path.join(OUT, tag)
    os.makedirs(od, exist_ok=True)
    cmd = ["lm_eval", "--model", "hf",
           "--model_args", f"pretrained={MODEL},dtype=float16",
           "--device", DEVICE,
           "--tasks", tasks, "--output_path", od, "--limit", LIMIT, "--batch_size", BATCH]
    if chat:
        cmd += ["--apply_chat_template"]
    if include_path:
        cmd += ["--include_path", include_path]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    print(f"\n>>>> {tag}: {tasks}  (chat_template={chat})", flush=True)
    subprocess.run(cmd, check=True)
    f = sorted(glob.glob(os.path.join(od, "**", "results*.json"), recursive=True),
               key=os.path.getmtime)[-1]
    return json.load(open(f))["results"]


def mean_metric(res, metric):
    vals = []
    for _, m in res.items():
        for k, v in m.items():
            if isinstance(v, (int, float)) and k.split(",")[0] == metric:
                vals.append(v)
                break
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    print(f"CALIBRATION  model={MODEL}  limit={LIMIT}/task  device={DEVICE}", flush=True)
    rows = []
    for label, ll_tasks, gen_tasks, inc, gk, gm, lm in SUITES:
        try:
            ll = run(ll_tasks, None, None, False, f"{label}_ll")        # canonical loglikelihood
            gen = run(gen_tasks, inc, gk, True, f"{label}_gen")         # Raidex generative
            a_ll, a_gen = mean_metric(ll, lm), mean_metric(gen, gm)
            rows.append((label, a_ll, a_gen))
            print(f"\n=== {label}: loglikelihood={a_ll:.4f}  generative={a_gen:.4f}  "
                  f"delta={a_gen - a_ll:+.4f} ===", flush=True)
        except Exception as e:
            print(f"\n!! {label} calibration failed: {str(e)[:200]}", flush=True)
            rows.append((label, None, None))

    print("\n#### CALIBRATION SUMMARY ####", flush=True)
    print(f"  model={MODEL}  n={LIMIT}/task", flush=True)
    for label, a_ll, a_gen in rows:
        if a_ll is None:
            print(f"  {label:7s}  (failed)", flush=True)
        else:
            print(f"  {label:7s}  loglikelihood={a_ll:.3f}  generative={a_gen:.3f}  "
                  f"delta={a_gen - a_ll:+.3f}", flush=True)
    print("#### CALIBRATION COMPLETE ####", flush=True)


if __name__ == "__main__":
    main()
