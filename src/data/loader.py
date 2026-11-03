from pathlib import Path

import pandas as pd


# Resolve the project root independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

REQUIRED_COLUMNS = ["id", "comment_text", *LABEL_COLUMNS]


def load_train_data(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and validate the Jigsaw training dataset.

    Args:
        path: Optional custom path to the training CSV.
              Defaults to data/raw/train.csv.

    Returns:
        A validated Pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns or valid values are missing.
    """

    csv_path = Path(path) if path is not None else RAW_DATA_DIR / "train.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate the dataset schema.
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Validate comment text.
    if df["comment_text"].isna().any():
        raise ValueError("Dataset contains missing comment text.")

    # Validate target labels.
    if df[LABEL_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing target labels.")

    invalid_labels = ~df[LABEL_COLUMNS].isin([0, 1])

    if invalid_labels.any().any():
        raise ValueError("Target labels must contain only 0 or 1.")

    return df
