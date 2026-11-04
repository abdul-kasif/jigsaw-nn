from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.split import split_train_validation
from src.preprocessing.text import build_vocabulary
from src.deep_learning.dataset import (
    ToxicCommentDataset,
    collate_fn,
)
from src.deep_learning.model import ToxicCommentBiLSTM

# --------------------------------------------------
# 1. Load Data and Build Vocabulary
# --------------------------------------------------

# Load the reproducible split.
train_df, val_df = split_train_validation()

# Build vocabulary using training comments only.
vocab = build_vocabulary(
    train_df["comment_text"],
    min_freq=2,
    max_vocab_size=30_000,
)

# Create datasets.
train_dataset = ToxicCommentDataset(
    train_df,
    vocab,
    max_length=300,
)

# Create DataLoader.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
)

# --------------------------------------------------
# 2. Get One Fixed Batch
# --------------------------------------------------

batch = next(iter(train_loader))

input_ids = batch["input_ids"]
lengths = batch["lengths"]
labels = batch["labels"]

print("Input shape:", input_ids.shape)
print("Labels shape:", labels.shape)
print("-" * 50)

# --------------------------------------------------
# 3. Initialize Model, Loss, and Optimizer
# --------------------------------------------------

model = ToxicCommentBiLSTM(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_size=128,
    num_labels=6,
    dropout=0.3,
    pad_id=vocab.pad_id,
)

criterion = nn.BCEWithLogitsLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.001,
)

# --------------------------------------------------
# 4. Observe learning over multiple updates (20 Steps)
# --------------------------------------------------

# Keep dropout disabled to isolate parameter updates.
model.eval()

loss_history = []

print("Starting 20-step optimization on a single batch...\n")

for step in range(20):
    # Forward pass
    logits = model(input_ids, lengths)

    # Calculate loss
    loss = criterion(logits, labels)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Step {step + 1:02d} | Loss: {loss.item():.6f}")

print("-" * 50)
print("Initial loss:", loss_history[0])
print("Final loss:", loss_history[-1])
