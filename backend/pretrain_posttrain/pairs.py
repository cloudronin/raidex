"""Model pair definitions for the pretraining vs post-training partition study.

Each entry pairs a base checkpoint with its instruct variant.

SERVING CONSTRAINT (verified 2026-06-26 against the HF Inference Providers API):
  Base/pretrained checkpoints CANNOT be reached through the existing Raidex pipeline.
  Both `_direct.complete` (chat-completions) and litellm `text_completion` route through
  HF's Inference Providers router, which rejects every base checkpoint with
  "is not a chat model" — gemma-3-27b-pt, gemma-2-27b, Llama-3.1-70B, Llama-3.1-8B,
  DeepSeek-V3-Base all fail this way. This is categorical (base models have no chat
  template), not an ID typo. Only instruct/chat-tagged models are served.

  Running base models therefore requires a TEXT-GENERATION serving path the current
  pipeline does not have: a dedicated GPU server (vLLM / TGI), an HF Dedicated
  Inference Endpoint, or another provider that exposes raw completions. The `base` IDs
  below are the canonical *existing* Hub checkpoints for that future path — verified to
  exist on the Hub, but NOT serverless-accessible today.

ALSO NOTE: Llama 3.3 shipped instruct-only — there is no Llama-3.3-70B base. The Llama
  family's clean same-version pair is Llama-3.1-70B (base) -> Llama-3.1-70B-Instruct.
"""

MODEL_PAIRS = [
    {
        "family": "Qwen2.5",
        "base":     "huggingface/Qwen/Qwen2.5-72B",
        "instruct": "huggingface/Qwen/Qwen2.5-72B-Instruct",
        "notes": "Clean same-version base/instruct pair (Qwen3-235B-A22B-Base does not exist "
                 "under that name; Qwen2.5-72B is a verified clean pair). Base not serverless.",
    },
    {
        "family": "Gemma-3",
        "base":     "huggingface/google/gemma-3-27b-pt",
        "instruct": "huggingface/google/gemma-3-27b-it",
        "notes": "Pretrained checkpoint is -pt (exists on Hub; serverless rejects 'not a chat model').",
    },
    {
        "family": "Llama-3.1",
        "base":     "huggingface/meta-llama/Llama-3.1-70B",
        "instruct": "huggingface/meta-llama/Llama-3.1-70B-Instruct",
        "notes": "Llama 3.3 is instruct-only (no base); 3.1-70B is the clean same-version pair. "
                 "Roster instruct (sambanova/Meta-Llama-3.3-70B-Instruct) is a different version.",
    },
    {
        "family": "DeepSeek-V3",
        "base":     "huggingface/deepseek-ai/DeepSeek-V3-Base",
        "instruct": "sambanova/DeepSeek-V3.2",
        "notes": "DeepSeek-V3-Base exists on Hub (pre-SFT) but serverless rejects 'not a chat model'. "
                 "671B MoE — needs heavy GPU to self-host.",
    },
]

# ---------------------------------------------------------------------------
# LOCAL pairs — Test 1 via local Ollama (the only viable base-model path; see the
# SERVING CONSTRAINT above). Small, clean SAME-VERSION base/instruct pairs, served
# identically through local Ollama so the base→instruct delta isolates post-training
# (not serving/precision differences). Both variants run locally; we do NOT compare a
# local base against the serverless board's instruct numbers (that would confound
# serving with training). Tags are 4-bit quantized — a fidelity compromise that
# cancels out because base and instruct use the same quant.
#
# model_id uses the openai/ prefix so litellm routes to Ollama's OpenAI-compatible
# endpoint when OPENAI_API_BASE=http://localhost:11434/v1 (set by run_local.py).
# Base tags: Llama/Gemma use "-text", Qwen uses "-base". Verified/adjusted at pull time.
LOCAL_PAIRS = [
    {
        "family":        "Llama-3.1-8B",
        "base_tag":      "llama3.1:8b-text-q4_K_M",
        "instruct_tag":  "llama3.1:8b-instruct-q4_K_M",
    },
    {
        "family":        "Gemma-2-9B",
        "base_tag":      "gemma2:9b-text-q4_K_M",
        "instruct_tag":  "gemma2:9b-instruct-q4_K_M",
    },
    {
        # Mistral AI as the 3rd distinct lab (Qwen2.5 ships no base on the Ollama library —
        # 404 on every -base/-text tag). Meta + Google + Mistral = the cross-family
        # diversity the kill criterion wants.
        "family":        "Mistral-7B",
        "base_tag":      "mistral:7b-text-q4_K_M",
        "instruct_tag":  "mistral:7b-instruct-q4_K_M",
    },
]


def local_model_id(tag: str) -> str:
    """litellm model_id for an Ollama tag, routed via the openai/ provider + OPENAI_API_BASE."""
    return "openai/" + tag


# Benchmarks that use an LLM judge to score responses.
# Base model completions for these are flagged format_confounded in the result JSON because:
# the task format (chat template + instruction following) may be unfamiliar to base models,
# so a low base score could reflect format confusion rather than genuine lack of alignment.
# Inspect the raw completions in base_sanity/ before interpreting the delta for these.
JUDGE_BENCHMARKS = {"strongreject", "xstest", "simpleqa"}

# Predicted classification per the study spec.
PRETRAINING_BOUND = {"wmdp", "simpleqa"}
POST_TRAINING_BOUND = {"strongreject", "xstest", "bbq", "ethics"}
