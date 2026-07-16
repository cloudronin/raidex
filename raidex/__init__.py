"""raidex — self-contained CLI to measure an LLM against the Raidex Responsible-AI index.

The pure eval-and-score core lives in ``raidex.core``; ``raidex.cli`` is the command-line
frontend. The backend service (queue poller + results-dataset upload) is a second, separate
frontend over the same core, so a local ``raidex eval`` score is identical in scale to the
published leaderboard.
"""

__version__ = "0.1.0"
