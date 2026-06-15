from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def run_dir(project: str, smoke: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("artifacts") / f"{stamp}-{'smoke' if smoke else 'full'}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def environment() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "uncommitted"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {key: clean(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [clean(item) for item in obj]
        if isinstance(obj, tuple):
            return [clean(item) for item in obj]
        if isinstance(obj, (np.integer, np.floating)):
            obj = obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    def default(obj):
        raise TypeError(type(obj).__name__)

    path.write_text(
        json.dumps(clean(value), indent=2, sort_keys=True, default=default, allow_nan=False) + "\n"
    )
