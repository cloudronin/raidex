"""Resolve the ``litellm`` / ``lm_eval`` console-scripts robustly across pip, venv, conda,
and pipx.

The lm-eval benchmarks spawn ``litellm`` (the OpenAI-compatible proxy) and ``lm_eval`` as
subprocesses *by name*. Under a plain venv/conda those are on PATH, but under **pipx** the
package's ``bin/`` is deliberately NOT on the user's PATH, so a bare ``Popen(["lm_eval",…])``
would fail. Resolve the script next to our own interpreter first (where a dependency's
console-script is installed), then fall back to PATH, then to the bare name.
"""
from __future__ import annotations

import os
import shutil
import sys


def exe(name: str) -> str:
    cand = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(cand):
        return cand
    return shutil.which(name) or name
