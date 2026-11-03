import torch
from torch.autograd import backward
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class ToxicCommentBiLSTM(nn.Module):
    """
    Bidirectional LSTM for multi-label toxic comment classification.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_labels: int = 6,
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()

        # Convert token IDs into learned dense vectors.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        # Process sequences in both directions.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Concatenated both forward and backward states.
        self.classifer = nn.Linear(
            hidden_size * 2,
            num_labels,
        )

    def forward(self, input_ids, lengths):
        # Step 1: Convert token IDs into embeddings.
        embedded = self.embedding(input_ids)

        # Step 2: Ignore padding during recurrent processing.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Step 3: Process the sequence with the BiLSTM.
        _, (hidden, cell) = self.lstm(packed)

        # Step 4: Extract the final forward and backward states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        # Step 5: Concatenate both directions.
        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        # Step 6: Apply dropout.
        combined = self.dropout(combined)

        # Step 7: Generate six classification logits.
        logits = self.classifer(combined)

        return logits
