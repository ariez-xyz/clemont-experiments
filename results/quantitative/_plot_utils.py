"""Shared utilities for quantitative result plotting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

_DEFAULT_PATTERN = "quant_run_*.json"


def resolve_json_paths(candidate: Optional[Path], *, default_dir: Path, pattern: str = _DEFAULT_PATTERN) -> List[Path]:
    """Return a list of JSON paths to process.

    If *candidate* is a file, return it. If *candidate* is a directory, search
    for files matching *pattern* inside it. When *candidate* is ``None`` use the
    provided *default_dir* instead. Raises ``SystemExit`` if nothing is found.
    """

    search_dir: Optional[Path]

    if candidate is None:
        search_dir = default_dir
    else:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            search_dir = candidate
        else:
            raise SystemExit(f"JSON file or directory not found: {candidate}")

    assert search_dir is not None
    json_files = sorted(search_dir.glob(pattern))
    if not json_files:
        raise SystemExit(f"No {pattern} files found in {search_dir}")
    return json_files


def metadata_value(meta: Sequence, key: str, fallback_key: Optional[str] = None):
    """Fetch ``meta[key]`` with optional fallback to ``meta[fallback_key]``.

    Works with both dicts and json-like mappings that implement ``get``.
    Returns ``None`` if keys are absent.
    """

    if isinstance(meta, dict):
        if key in meta:
            return meta[key]
        if fallback_key is not None and fallback_key in meta:
            return meta[fallback_key]
        return None

    try:
        return meta[key]
    except (KeyError, TypeError):
        if fallback_key is None:
            return None
        try:
            return meta[fallback_key]
        except (KeyError, TypeError):
            return None

