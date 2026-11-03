from sklearn.model_selection import train_test_split

from src.data.loader import load_train_data

from pathlib import Path

import numpy as np


def split_train_validation(
    validation_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the training dataset into training and validation sets.

    Returns:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
    """

    df = load_train_data()

    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_df, val_df


def save_split_indices(
    train_df,
    val_df,
    path: str | Path | None = None,
):
    """Save the row indices used for the train/validation split."""

    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "processed"
            / "split_indices.npz"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        train_indices=train_df.index.to_numpy(),
        val_indices=val_df.index.to_numpy(),
    )
