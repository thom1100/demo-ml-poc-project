"""Feature engineering utilities for the bike counters project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the input feature dataframe."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "build_features expects a pandas.DataFrame as input."
        )

    if "date" not in X.columns:
        raise ValueError(
            "Input dataframe must contain a 'date' column."
        )

    X = X.copy()
    X["date"] = pd.to_datetime(X["date"], errors="coerce")

    if X["date"].isna().any():
        raise ValueError(
            "Some values in the 'date' column could not be parsed as datetimes."
        )

    return X


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from the raw dataframe.

    Parameters
    ----------
    X:
        Raw feature dataframe. Must contain a ``date`` column.

    Returns
    -------
    pd.DataFrame
        A new dataframe containing engineered features.

    Notes
    -----
    This function:
    - extracts calendar-based features from ``date``
    - encodes cyclical time information with sine/cosine transforms
    - preserves any existing non-date columns
    - drops the original ``date`` column at the end
    """
    X = _validate_input(X)

    features = X.copy()

    # Calendar features
    features["year"] = features["date"].dt.year
    features["month"] = features["date"].dt.month
    features["day"] = features["date"].dt.day
    features["hour"] = features["date"].dt.hour
    features["day_of_week"] = features["date"].dt.dayofweek
    features["week_of_year"] = features["date"].dt.isocalendar().week.astype(int)

    # Binary calendar flags
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for periodic features
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)

    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    # Drop raw datetime column once derived features are created
    features = features.drop(columns=["date"])

    return features