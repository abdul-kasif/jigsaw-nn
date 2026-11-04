import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
    multilabel_confusion_matrix,
)

from src.data.loader import LABEL_COLUMNS


def to_numpy(data):
    """Convert a PyTorch tensor or NumPy array to NumPy."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return np.asarray(data)


def calculate_metrics(
    y_true,
    y_prob,
    threshold=0.5,
    label_names=LABEL_COLUMNS,
):
    """
    Calculate evaluation metrics for a multi-label classifier.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels. Shape: (samples, labels).

    y_prob:
        Predicted probabilities. Shape: (samples, labels).

    threshold:
        Probability threshold used to generate binary predictions.

    label_names:
        Names of the classification labels.
    """

    y_true = to_numpy(y_true)
    y_prob = to_numpy(y_prob)

    # Validate input dimensions.
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true and y_prob must be 2D arrays.")

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )

    if y_true.shape[1] != len(label_names):
        raise ValueError("Number of labels does not match label_names.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Predicted probabilities contain NaN or infinity.")

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Ground-truth labels must contain only 0 and 1.")

    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics separately for each label.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label = pd.DataFrame(
        {
            "label": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    # Calculate aggregate metrics.
    aggregate = {
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Calculate one 2x2 confusion matrix per label.
    confusion_matrices = multilabel_confusion_matrix(
        y_true,
        y_pred,
    )

    confusion_matrices = {
        label: matrix for label, matrix in zip(label_names, confusion_matrices)
    }

    return {
        "per_label": per_label,
        "aggregate": aggregate,
        "confusion_matrices": confusion_matrices,
    }
