"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FILE = DATA_DIR / "raw/bikes.parquet"


def _get_train_path() -> Path:
    """Return the expected path to the training parquet file."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Expected location: <project_root>/data/train.parquet"
        )
    return TRAIN_FILE


def _load_train_dataframe() -> pd.DataFrame:
    """Load the raw training dataframe."""
    df = pd.read_parquet(_get_train_path())

    if "date" not in df.columns:
        raise ValueError(
            "The training dataframe must contain a 'date' column."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(
            "Some values in the 'date' column could not be parsed as datetimes."
        )

    return df.sort_values("date").reset_index(drop=True)


def _build_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target.

    Target priority:
    'log_bike_count' if already available

    The returned target is always named 'bike_log_count' for consistency.
    """
    df = df.copy()

    if "log_bike_count" in df.columns:
        y = df["log_bike_count"].copy()
        target_cols_to_drop = ["log_bike_count"]

    else:
        raise ValueError(
            "No target column found. Expected 'log_bike_count'"
        )

    y = y.rename("bike_log_count")

    columns_to_drop = list(target_cols_to_drop)

    # Drop leakage-prone identifier if it exists and is not meant to be a feature.

    if "Id" in df.columns:
        columns_to_drop.append("Id")

    X = df.drop(columns=columns_to_drop, errors="ignore")

    return X, y


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    delta_threshold: str = "30 days",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a temporal train/test split.

    The most recent period is used as the test set.
    """
    if "date" not in X.columns:
        raise ValueError("X must contain a 'date' column for temporal splitting.")

    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    train_mask = X["date"] <= cutoff_date

    if train_mask.all() or (~train_mask).all():
        raise ValueError(
            "Temporal split failed: one of the splits is empty. "
            "Check the date range in the dataset."
        )

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[~train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[~train_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """
    df = _load_train_dataframe()
    X, y = _build_target(df)
    return _temporal_split(X, y)