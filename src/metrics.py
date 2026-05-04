"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got {y_true_array.shape} and {y_pred_array.shape}."
        )

    rmse = root_mean_squared_error(y_true_array, y_pred_array)
    mae = mean_absolute_error(y_true_array, y_pred_array)
    r2 = r2_score(y_true_array, y_pred_array)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
