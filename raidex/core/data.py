"""raidex.core.data — unified benchmark-data cache, version pinning, and offline mode.

Every benchmark loader resolves its data through here so the package has ONE cache dir
(not scattered ``/tmp`` paths), pins each source to a fixed version for reproducibility,
and supports a hard offline mode for air-gapped use. ``raidex fetch-data`` pre-populates
the cache; ``--offline`` then runs with zero network (a cache miss is a clear, actionable
error, not a stack trace). For air-gap: run ``fetch-data`` on a networked machine and copy
``data_dir()`` across.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

_OFFLINE = False


def data_dir() -> Path:
    """The single cache root: ``$RAIDEX_DATA_DIR`` or the platform user-cache dir."""
    d = os.environ.get("RAIDEX_DATA_DIR")
    if d:
        p = Path(d)
    else:
        try:
            from platformdirs import user_cache_dir
            p = Path(user_cache_dir("raidex")) / "data"
        except Exception:
            p = Path.home() / ".cache" / "raidex" / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_offline() -> bool:
    return _OFFLINE or os.environ.get("RAIDEX_OFFLINE") == "1"


def set_offline(on: bool = True) -> None:
    """Enable hard offline mode: no network, HF hub/datasets pinned to the local cache."""
    global _OFFLINE
    _OFFLINE = on
    if on:
        os.environ["RAIDEX_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        for k in ("RAIDEX_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(k, None)
    # Route the HF cache into our data dir so fetch-data and offline share one location.
    os.environ.setdefault("HF_HOME", str(data_dir() / "hf"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_file(name: str, url: str, sha256: str | None = None) -> Path:
    """Local path for ``name``, fetched once from ``url`` (verified against ``sha256``) and
    cached under ``data_dir()``. Offline + cache-miss raises an actionable error."""
    dest = data_dir() / name
    if dest.exists():
        return dest
    if is_offline():
        raise RuntimeError(
            f"offline: '{name}' is not in the cache ({dest}). Run `raidex fetch-data` on a "
            f"networked machine and copy {data_dir()} across."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256:
        got = _sha256(tmp)
        if got != sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: sha256 mismatch (expected {sha256}, got {got})")
    tmp.replace(dest)
    return dest


def hf_dataset(repo: str, *, name: str | None = None, split: str | None = None,
               revision: str | None = None):
    """``datasets.load_dataset`` pinned to ``revision`` and routed through our cache;
    honors offline via the HF_*_OFFLINE env set by ``set_offline``."""
    os.environ.setdefault("HF_HOME", str(data_dir() / "hf"))
    from datasets import load_dataset
    return load_dataset(repo, name, split=split, revision=revision)


# ---------------------------------------------------------------------------------------
# Pinned versions — captured at build time so scores are reproducible and comparable over
# time. HF repos pin by commit sha; direct CSV/txt sources pin by sha256. `revision=None`
# / `sha256=None` means "not yet pinned" (tracks upstream); `raidex fetch-data --record`
# refreshes these. See raidex.core.data.DATASET_VERSIONS consumers in each benchmark.
# ---------------------------------------------------------------------------------------
DATASET_VERSIONS: dict[str, dict] = {
    "bbq":          {"source": "oskarvanderwal/bbq", "revision": "ab00114b0fbf59c7c539ff9158f6ce717d13ab63"},
    "wmdp":         {"source": "cais/wmdp", "revision": "7125571f22f032c56415e7980f48d877dd830ff8"},
    "ethics":       {"source": "lighteval/hendrycks_ethics", "revision": "b7644ea36eb8ce24858fff602ff47762d39bc3b5"},
    "xstest":       {"source": "Paul/XSTest", "revision": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5"},
    "advglue":      {"source": "AI-Secure/adv_glue", "revision": "e1abda026f687e917a8d9895469194736ebe872c"},
    "sycophancy":   {"source": "meg-tong/sycophancy-eval", "revision": "18f18160e9998e524bbf027c4475bfa7fdeb6949"},
    "simpleqa":     {"source": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
                     "sha256": "feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032"},
    "confaide":     {"source": "https://raw.githubusercontent.com/skywalker023/confaide/main/benchmark",
                     "sha256": {"tier_1.txt": "63261cf847efb3f35f359913bea828131298ee04980189ed7d75eae05f76ac58",
                                "tier_2a.txt": "a869099fab404cde9dd50f23c20668063021afa8f080e190b348a5a4da14b979"}},
    "strongreject": {"source": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
                     "sha256": "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"},
}


def pin(bid: str) -> dict:
    """The pinned-version record for a benchmark id (source + revision|sha256)."""
    return DATASET_VERSIONS.get(bid, {"source": None})


def revision(bid: str) -> str | None:
    return DATASET_VERSIONS.get(bid, {}).get("revision")
