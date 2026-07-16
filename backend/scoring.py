"""Compatibility shim — the scoring core moved to ``raidex.core.scoring``.

Kept so the backend service (``runner.py --poll``) and the ``pretrain_posttrain``
research scripts keep working with an unmodified ``import scoring`` (run with cwd=backend).
Remove once every caller imports ``raidex.core`` directly.
"""
from raidex.core.scoring import *  # noqa: F401,F403
