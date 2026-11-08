"""Splits, metrics and the experiment log."""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .types import F64

VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"
RESULTS = Path("results.csv")


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    played = df[df.home_win.notna()]
    train = played[played.SEASON < VAL_SEASON]
    return train, played[played.SEASON == VAL_SEASON], played[played.SEASON == TEST_SEASON]


def metrics(y: F64, p: F64) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "acc": float(np.mean((p > 0.5) == (y == 1))),
        "ece": float(ece),
        "n": float(len(y)),
    }


def _sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def log(model: str, variant: str, split_name: str, m: dict[str, float], note: str = "") -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "sha", "model", "variant", "split", "logloss", "brier", "acc", "ece", "n", "note"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), _sha(), model, variant, split_name,
                    f"{m['logloss']:.4f}", f"{m['brier']:.4f}", f"{m['acc']:.4f}", f"{m['ece']:.4f}", int(m["n"]), note])


CV_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]


def walk_forward(df: pd.DataFrame, seasons: list[str] = CV_SEASONS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """(train, holdout) pairs: train on every played game before each holdout season."""
    played = df[df.home_win.notna()]
    return [(played[played.SEASON < s], played[played.SEASON == s]) for s in seasons]
