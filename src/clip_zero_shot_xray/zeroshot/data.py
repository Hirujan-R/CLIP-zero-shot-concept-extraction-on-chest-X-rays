"""CheXpert data access helpers (path remapping, labels, sampling)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prompts import ALL_CONCEPTS

PATH_PREFIX = "CheXpert-v1.0-small/"


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk up from ``start`` (or cwd) until the CheXpert raw data is found."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "01_raw" / "chexpert" / "valid.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate data/01_raw/chexpert/valid.csv above "
        f"{current!s}"
    )


def valid_raw_dir(root: Path | str | None = None) -> Path:
    """Return the directory holding the CheXpert raw files."""
    base = Path(root).resolve() if root else find_repo_root()
    return base / "data" / "01_raw" / "chexpert"


def load_valid(
    root: Path | str | None = None, frontal_only: bool = True
) -> pd.DataFrame:
    """Load ``valid.csv`` with local image paths and (optionally) frontal views."""
    raw = valid_raw_dir(root)
    df = pd.read_csv(raw / "valid.csv")
    df["image_path"] = (
        df["Path"].str.replace(PATH_PREFIX, "", regex=False).map(lambda p: raw / p)
    )
    if frontal_only:
        df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    return df


def labels_matrix(
    df: pd.DataFrame, concepts: list[str] | None = None
) -> pd.DataFrame:
    """Return the binary label matrix restricted to known concept columns."""
    concepts = concepts or ALL_CONCEPTS
    concepts = [c for c in concepts if c in df.columns]
    return df[concepts].astype(float).copy()


def metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return acquisition/patient metadata used for subgroup analysis."""
    meta = df[["Sex", "Age", "AP/PA"]].copy()
    meta["age_bin"] = pd.cut(
        meta["Age"],
        bins=[0, 40, 60, 80, 120],
        labels=["<40", "40-59", "60-79", "80+"],
    )
    return meta


def sample_images(
    df: pd.DataFrame, n: int = 8, seed: int = 0
) -> pd.DataFrame:
    """Return up to ``n`` randomly sampled rows (deterministic for a seed)."""
    n = max(1, min(n, len(df)))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
