from functools import partial

from torch.utils.data import DataLoader

from src.data.split import split_train_validation
from src.preprocessing.text import build_vocabulary
from src.deep_learning.dataset import ToxicCommentDataset, collate_fn


# 1. Split the dataset.
train_df, val_df = split_train_validation()

# 2. Build vocabulary using training comments only.
vocab = build_vocabulary(
    train_df["comment_text"],
    min_freq=2,
    max_vocab_size=30_000,
)

# 3. Create training dataset.
train_dataset = ToxicCommentDataset(
    train_df,
    vocab,
    max_length=300,
)

# 4. Create validation dataset.
val_dataset = ToxicCommentDataset(
    val_df,
    vocab,
    max_length=300,
)

# 5. Create training DataLoader.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
)

# 6. Create validation DataLoader.
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Vocabulary size:", len(vocab))

# Get one training batch.
train_batch = next(iter(train_loader))

# Get one validation batch.
val_batch = next(iter(val_loader))

print("TRAINING BATCH")
print("Input IDs:", train_batch["input_ids"].shape)
print("Labels:", train_batch["labels"].shape)

print("\nVALIDATION BATCH")
print("Input IDs:", val_batch["input_ids"].shape)
print("Labels:", val_batch["labels"].shape)
