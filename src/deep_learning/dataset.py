from pandas.io.parquet import to_parquet
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data.loader import LABEL_COLUMNS
from src.preprocessing.text import Vocabulary, tokenize


class ToxicCommentDataset(Dataset):
    """
    PyTorch Dataset for multi-label toxic comment classification.
    """

    def __init__(
        self,
        dataframe,
        vocabulary: Vocabulary,
        max_length: int = 300,
    ):

        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.texts = dataframe["comment_text"].tolist()

        self.labels = dataframe[LABEL_COLUMNS].to_numpy(dtype="float32")

        self.vocabulary = vocabulary

        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Step 1: Get the original comment.
        text = self.texts[index]

        # Step 2: Convert text into tokens.
        tokens = tokenize(text)

        # Step 3: Covert tokens into integer IDs.
        token_ids = self.vocabulary.encode(tokens)

        # Step 4: Truncate excessively long comments.
        token_ids = token_ids[: self.max_length]

        # Step 5: Convert input and lables into tensors.
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )

        labels = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch, pad_id=0):
    """
    Combine individual samples into a padded batch.

    Sequences are padded to the longest sequence in the batch.
    """

    input_sequences = [sample["input_ids"] for sample in batch]

    labels = torch.stack([sample["labels"] for sample in batch])

    lengths = torch.tensor(
        [len(sequence) for sequence in input_sequences],
        dtype=torch.long,
    )

    padded_sequences = pad_sequence(
        input_sequences,
        batch_first=True,
        padding_value=pad_id,
    )

    attention_mask = padded_sequences != pad_id

    return {
        "input_ids": padded_sequences,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": labels,
    }
