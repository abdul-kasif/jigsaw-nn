# Experiment 02: Class-Weighted BiLSTM for Toxic Comment Classification

**Project:** Jigsaw Toxic Comment Classification  
**Experiment:** 02  
**Model:** Bidirectional LSTM (BiLSTM)  
**Experiment status:** Completed  
**Best checkpoint:** `models/bilstm_ex2_best.pt`  
**Notebook:** `notebooks/model_training_ex2.ipynb`

---

## 1. Objective

The objective of this experiment was to investigate whether class weighting could improve the BiLSTM model's ability to identify minority classes in the highly imbalanced Jigsaw Toxic Comment Classification dataset.

In Experiment 01, the baseline BiLSTM used standard Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss). At the default classification threshold of 0.5, the model struggled to identify the minority labels:

- `threat`: F1 = 0.0000
- `identity_hate`: F1 = 0.0000
- `severe_toxic`: F1 = 0.2278

The model frequently predicted these labels as negative, resulting in poor minority-class recall.

**Hypothesis:**

Using class-weighted BCEWithLogitsLoss will encourage the model to pay greater attention to positive examples from underrepresented classes and improve minority-label detection.

## 2. Experimental Changes

Experiment 02 retains the same BiLSTM architecture and data preparation pipeline as Experiment 01.

The primary change is the introduction of class-specific positive weights in the loss function.

| Component               | Experiment 01              | Experiment 02              |
| ----------------------- | -------------------------- | -------------------------- |
| Model architecture      | BiLSTM                     | BiLSTM (unchanged)         |
| Embedding dimension     | 100                        | 100                        |
| Hidden size             | 128                        | 128                        |
| Bidirectional           | Yes                        | Yes                        |
| Dropout                 | 0.3                        | 0.3                        |
| Output labels           | 6                          | 6                          |
| Loss function           | Standard BCEWithLogitsLoss | Weighted BCEWithLogitsLoss |
| Optimizer               | Adam                       | Adam                       |
| Learning rate           | 0.001                      | 0.001                      |
| Batch size              | 32                         | 32                         |
| Maximum sequence length | 300                        | 300                        |
| Training epochs         | 1                          | 5                          |
| Validation split        | 20%                        | Same fixed split           |
| Random seed             | 42                         | 42                         |
| Vocabulary size         | 30,000                     | 30,000                     |

The model contains **3,237,062 trainable parameters**.

The architecture, tokenizer, vocabulary, sequence-length configuration, dataset, and train/validation split were kept unchanged to isolate the effect of class weighting as much as possible.

The Experiment 02 model was initialized from scratch. It did not continue training from the Experiment 01 checkpoint.

## 3. Dataset Class Imbalance Analysis

Class weights were calculated using only the training split, which contains 127,656 samples.

The positive weight for each label was calculated using:

$$w_j = \frac{N_{\text{negative},j}}{N_{\text{positive},j}}$$

To prevent extremely large weights from dominating the optimization process, a maximum weight of 20 was applied:

$$w_j^{\text{applied}} = \min(w_j,20)$$

### Calculated class weights

| Label         | Positive samples | Negative samples | Raw weight | Applied weight |
| ------------- | ---------------: | ---------------: | ---------: | -------------: |
| toxic         |           12,238 |          115,418 |       9.43 |           9.43 |
| severe_toxic  |            1,274 |          126,382 |      99.20 |          20.00 |
| obscene       |            6,734 |          120,922 |      17.96 |          17.96 |
| threat        |              404 |          127,252 |     314.98 |          20.00 |
| insult        |            6,263 |          121,393 |      19.38 |          19.38 |
| identity_hate |            1,111 |          126,545 |     113.90 |          20.00 |

The three most imbalanced labels, `severe_toxic`, `threat`, and `identity_hate`, reached the maximum weight of 20.

### Implementation

```python
positive_counts = train_df[LABEL_COLUMNS].sum().to_numpy()
negative_counts = len(train_df) - positive_counts

raw_weights = negative_counts / positive_counts

MAX_POS_WEIGHT = 20.0

pos_weights = np.clip(
    raw_weights,
    a_min=None,
    a_max=MAX_POS_WEIGHT
)

pos_weight_tensor = torch.tensor(
    pos_weights,
    dtype=torch.float32,
    device=device
)
```

The weighted loss was implemented as:

```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight_tensor
)
```

The positive examples receive a larger loss contribution according to their respective class weights. Negative examples are not scaled by `pos_weight`.

## 4. Training Configuration

The model was trained for five epochs using the following configuration:

```python
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_POS_WEIGHT = 20.0

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
```

Training was performed using CUDA on an NVIDIA GeForce RTX 2050.

### Training results

| Epoch | Training loss | Validation loss | Training time |
| ----: | ------------: | --------------: | ------------: |
|     1 |      0.389525 |        0.288217 |       65.48 s |
|     2 |      0.234610 |        0.243643 |       65.31 s |
|     3 |      0.181321 |    **0.224035** |       66.30 s |
|     4 |      0.144707 |        0.252793 |       66.70 s |
|     5 |      0.117681 |        0.307338 |       68.80 s |

### Training observations

Training loss decreased continuously across all five epochs.

Validation loss decreased until Epoch 3, reaching its minimum of 0.224035. It subsequently increased during Epochs 4 and 5.

This indicates that the model began overfitting after Epoch 3.

Epoch 3 was therefore selected as the best checkpoint based on minimum validation loss.

### Checkpoint implementation

The model checkpoint was saved whenever validation loss improved.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_epoch = epoch + 1

    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
        },
        checkpoint_path,
    )
```

**Selected checkpoint:** `models/bilstm_ex2_best.pt`

**Best epoch:** 3

**Best validation loss:** 0.224035

The saved checkpoint was subsequently loaded before evaluation.

## 5. Evaluation at the Default Threshold (0.5)

The selected Epoch 3 checkpoint was initially evaluated using the standard classification threshold of 0.5.

### Per-label evaluation

| Label         | Precision |   Recall |       F1 | Support |
| ------------- | --------: | -------: | -------: | ------: |
| toxic         |  0.567469 | 0.917866 | 0.701338 |   3,056 |
| severe_toxic  |  0.240233 | 0.900312 | 0.379265 |     321 |
| obscene       |  0.530705 | 0.957434 | 0.682886 |   1,715 |
| threat        |  0.151420 | 0.648649 | 0.245524 |      74 |
| insult        |  0.439702 | 0.951053 | 0.601371 |   1,614 |
| identity_hate |  0.134355 | 0.775510 | 0.229031 |     294 |

### Aggregate evaluation

| Metric          |  Score |
| --------------- | -----: |
| Micro precision | 0.4440 |
| Micro recall    | 0.9255 |
| Micro F1        | 0.6001 |
| Macro F1        | 0.4732 |
| Weighted F1     | 0.6350 |

### Observations

Class weighting significantly increased recall.

For example:

- `threat` achieved a recall of 0.648649.
- `identity_hate` achieved a recall of 0.775510.
- `severe_toxic` achieved a recall of 0.900312.

However, precision decreased substantially. The model generated many false-positive predictions.

This indicates that the default threshold of 0.5 was not suitable for the probability distributions produced by the weighted model.

## 6. Per-Label Threshold Optimization

Since class weighting changed the model's prediction behaviour, individual classification thresholds were investigated.

Thresholds from 0.05 to 0.95 were initially tested in increments of 0.05.

The best-performing thresholds were then investigated more closely using an extended sweep from 0.80 to 0.99 in increments of 0.01.

The best thresholds found were:

| Label         | Selected threshold |     F1 |
| ------------- | -----------------: | -----: |
| toxic         |               0.88 | 0.7988 |
| severe_toxic  |               0.93 | 0.5103 |
| obscene       |               0.92 | 0.7994 |
| threat        |               0.81 | 0.3492 |
| insult        |               0.89 | 0.7157 |
| identity_hate |               0.85 | 0.3286 |

### Threshold selection implementation

```python
extended_thresholds = np.arange(
    0.80,
    1.00,
    0.01
)

for i, label in enumerate(LABEL_COLUMNS):
    scores = []

    for threshold in extended_thresholds:
        predictions = (
            y_prob_np[:, i] >= threshold
        ).astype(int)

        score = f1_score(
            y_true_np[:, i],
            predictions,
            zero_division=0
        )

        scores.append(score)

    best_idx = np.argmax(scores)
```

### Updated per-label evaluation

| Label         | Threshold | Precision | Recall |     F1 |
| ------------- | --------: | --------: | -----: | -----: |
| toxic         |      0.88 |    0.8038 | 0.7938 | 0.7988 |
| severe_toxic  |      0.93 |    0.4352 | 0.6168 | 0.5103 |
| obscene       |      0.92 |    0.7644 | 0.8379 | 0.7994 |
| threat        |      0.81 |    0.2870 | 0.4459 | 0.3492 |
| insult        |      0.89 |    0.6424 | 0.8079 | 0.7157 |
| identity_hate |      0.85 |    0.2816 | 0.3946 | 0.3286 |

### Aggregate evaluation with optimized thresholds

| Metric          |  Score |
| --------------- | -----: |
| Micro precision | 0.6971 |
| Micro recall    | 0.7795 |
| Micro F1        | 0.7360 |
| Macro F1        | 0.5837 |
| Weighted F1     | 0.7427 |

Threshold optimization improved the balance between precision and recall, resulting in a higher F1 score across all aggregate metrics.

## 7. Comparison with Experiment 01

Experiment 01 used a standard unweighted BCEWithLogitsLoss and a single threshold of 0.5.

Experiment 02 used class-weighted BCEWithLogitsLoss and individually optimized thresholds.

### Aggregate comparison

| Metric          | Experiment 01 | Experiment 02 (threshold 0.5) | Experiment 02 (optimized) |
| --------------- | ------------: | ----------------------------: | ------------------------: |
| Micro precision |        0.8463 |                        0.4440 |                    0.6971 |
| Micro recall    |        0.5567 |                        0.9255 |                    0.7795 |
| Micro F1        |        0.6716 |                        0.6001 |                **0.7360** |
| Macro F1        |        0.3880 |                        0.4732 |                **0.5837** |
| Weighted F1     |        0.6502 |                        0.6350 |                **0.7427** |

### Per-label F1 comparison

| Label         | Experiment 01 (0.5) | Experiment 02 (optimized) |
| ------------- | ------------------: | ------------------------: |
| toxic         |              0.7365 |                    0.7988 |
| severe_toxic  |              0.2278 |                    0.5103 |
| obscene       |              0.7376 |                    0.7994 |
| threat        |              0.0000 |                    0.3492 |
| insult        |              0.6261 |                    0.7157 |
| identity_hate |              0.0000 |                    0.3286 |

All six labels improved in F1 compared with Experiment 01.

The largest improvements were observed in the minority labels:

- `severe_toxic`: +0.2825
- `threat`: +0.3492
- `identity_hate`: +0.3286

### Interpretation

The results support the hypothesis that class weighting can improve the model's ability to identify underrepresented classes.

However, class weighting alone did not improve every aggregate metric at the default threshold. Threshold optimization was important in obtaining the final performance.

The macro F1 improvement is particularly significant because macro F1 gives equal importance to all six labels.

## 8. Limitations and Considerations

Several limitations should be considered when interpreting these results.

### 8.1 Validation-based threshold selection

The individual thresholds were selected using the validation set. The same validation set was also used for checkpoint selection.

Consequently, the optimized validation scores may be optimistic and should not be interpreted as unbiased estimates of generalization performance.

A separate calibration or threshold-selection split would provide a more rigorous evaluation.

### 8.2 Overfitting

Validation loss increased after Epoch 3, despite continued decreases in training loss.

This indicates that additional regularization or improved training strategies may be worth investigating in future experiments.

### 8.3 Remaining minority-class precision challenges

Although minority-label F1 improved, precision remains relatively low for:

- `threat`: 0.2870
- `identity_hate`: 0.2816

Reducing false positives while retaining reasonable recall remains an important challenge.

### 8.4 Threshold sensitivity

Different labels require different decision thresholds.

A single threshold is not necessarily appropriate for all six labels, especially when class weighting is used.

### 8.5 Weight cap

The maximum positive weight of 20 was a heuristic experimental choice. Other weight caps may produce different results and could be investigated in future experiments.

## 9. Conclusion

Experiment 02 successfully demonstrated that class-weighted BCEWithLogitsLoss can improve minority-class detection in the Jigsaw Toxic Comment Classification task.

Compared with the initial BiLSTM baseline, the optimized Experiment 02 model achieved:

- Micro F1: 0.7360
- Macro F1: 0.5837
- Weighted F1: 0.7427

The macro F1 increased from 0.3880 to 0.5837, demonstrating a substantial improvement in performance across the six labels.

Most importantly, the model learned to identify `threat` and `identity_hate`, which the baseline model failed to predict at the default threshold.

The experiment also demonstrated that threshold optimization is important when using class-weighted loss functions.

**Final conclusion:** Class weighting combined with per-label threshold optimization is a promising approach for addressing class imbalance in multi-label toxic comment classification.

## 10. Reproducibility and Artifacts

The following artifacts preserve the implementation and experimental evidence:

| Artifact                             | Purpose                                        |
| ------------------------------------ | ---------------------------------------------- |
| `notebooks/model_training_ex2.ipynb` | Training, evaluation, and threshold analysis   |
| `models/bilstm_ex2_best.pt`          | Best Epoch 3 model checkpoint                  |
| `src/deep_learning/model.py`         | BiLSTM architecture                            |
| `src/deep_learning/trainer.py`       | Training, evaluation, and prediction functions |
| `src/evaluation/metrics.py`          | Classification metric calculations             |
| `src/data/split.py`                  | Fixed training/validation split                |
| `src/preprocessing/text.py`          | Tokenization and vocabulary construction       |
| `data/processed/split_indices.npz`   | Saved split indices                            |

### Final experimental configuration

```python
EXPERIMENT_02_CONFIG = {
    "model": "BiLSTM",
    "embedding_dim": 100,
    "hidden_size": 128,
    "num_labels": 6,
    "dropout": 0.3,
    "loss": "BCEWithLogitsLoss",
    "max_pos_weight": 20.0,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 5,
    "best_epoch": 3,
    "best_validation_loss": 0.224035,
    "thresholds": {
        "toxic": 0.88,
        "severe_toxic": 0.93,
        "obscene": 0.92,
        "threat": 0.81,
        "insult": 0.89,
        "identity_hate": 0.85,
    },
}
```

**Experiment 02 status: COMPLETED**

The next experiment will investigate Focal Loss as an alternative approach to handling difficult and imbalanced examples.
