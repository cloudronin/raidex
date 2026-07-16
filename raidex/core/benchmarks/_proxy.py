"""litellm proxy lifecycle for lm-eval-harness benchmarks.

lm-eval talks to an OpenAI-compatible endpoint; we expose any litellm-supported
model as one by running ``litellm --model <id> --port <p>`` and pointing lm-eval
at it. The proxy serves the model under the *full* model_id string (e.g.
``openai/gpt-5.2``) — that is the name lm-eval must use, not the split tail.
Readiness is polled instead of a fixed sleep to avoid the cold-start race.
"""
from __future__ import annotations

import contextlib
import subprocess
import time
import urllib.request


@contextlib.contextmanager
def litellm_proxy(model_id: str, port: int | None = None, timeout_s: int = 90):
    """Start a litellm proxy for ``model_id``, yield ``(base_url, served_name)``,
    tear it down on exit. ``served_name == model_id``. Port defaults to
    RAIDEX_PROXY_PORT (else 8000) so concurrent runs can use distinct ports."""
    import os
    if port is None:
        port = int(os.environ.get("RAIDEX_PROXY_PORT", "8000"))
    os.makedirs("/tmp/raidex", exist_ok=True)
    log = open(f"/tmp/raidex/litellm_proxy_{port}.log", "w")
    proc = subprocess.Popen(
        # --drop_params: silently drop provider-unsupported params (e.g. Anthropic
        # rejects `seed`, which lm-eval sends) instead of 400-ing the request.
        ["litellm", "--model", model_id, "--port", str(port), "--drop_params"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        _wait_ready(port, timeout_s)
        yield f"http://localhost:{port}/v1", model_id
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        log.close()


def _wait_ready(port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        for path in ("/health/liveliness", "/v1/models"):
            try:
                with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as e:  # any failure just means "not ready yet"
                last_err = e
        time.sleep(1)
    raise RuntimeError(f"litellm proxy not ready on port {port} within {timeout_s}s: {last_err}")
