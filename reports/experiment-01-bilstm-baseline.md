
# Experiment 01 — BiLSTM Baseline for Toxic Comment Classification

**Project:** Jigsaw Neural Network  
**Experiment:** 01 — Initial BiLSTM baseline  
**Date:** 4 November 2026  
**Status:** Completed  
**Notebook:** `notebooks/model_training_ex1.ipynb`

---

## 1. Objective

The objective of this experiment was to implement and evaluate an initial deep learning pipeline for the Jigsaw Toxic Comment Classification task.

The experiment focused on:

- Building a text preprocessing and vocabulary pipeline.
- Converting comments into PyTorch tensors.
- Implementing a Bidirectional LSTM (BiLSTM) for multi-label classification.
- Training and evaluating the model using GPU acceleration.
- Establishing a neural network baseline.
- Investigating the impact of class imbalance and prediction thresholds.

This experiment establishes the initial reference point for subsequent improvements.

## 2. Dataset

The project uses the Kaggle Jigsaw Toxic Comment Classification dataset.

The task is a multi-label classification problem with six independent target labels:

1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

A single comment can belong to multiple categories simultaneously.

The training dataset contains 159,571 comments.

### Training and validation split

The dataset was divided into:

| Dataset | Samples |
| --- | ---: |
| Training | 127,656 |
| Validation | 31,915 |
| Total | 159,571 |

An 80/20 split was used with a fixed random state of 42.

The split indices were serialized to `data/processed/split_indices.npz` to support reproducible experiments and consistent comparisons with the classical machine learning baseline.

## 3. Data preprocessing and representation

### 3.1 Tokenization

A custom regular-expression tokenizer was implemented.

Its functionality includes:

- Recognizing URLs.
- Preserving words and contractions.
- Separating punctuation into individual tokens.
- Supporting Unicode word characters.

### 3.2 Vocabulary construction

A custom `Vocabulary` class was implemented to map tokens to integer IDs and integer IDs back to tokens.

Configuration:

| Parameter | Value |
| --- | ---: |
| Minimum token frequency | 2 |
| Maximum vocabulary size | 30,000 |
| Padding token | `<pad>` |
| Unknown token | `<unk>` |

The vocabulary was constructed using training comments only. The validation dataset reused the same vocabulary.

This prevents information from the validation text from influencing vocabulary construction.

### 3.3 Sequence preparation

Comments were converted into integer sequences.

| Parameter | Value |
| --- | ---: |
| Maximum sequence length | 300 |
| Batch size | 32 |
| Padding strategy | Dynamic batch padding |
| Label representation | Float32 |
| Input representation | Integer token IDs |

Comments exceeding 300 tokens were truncated. Sequences within a batch were dynamically padded using the custom `collate_fn`.

Sequence lengths were retained separately for packed LSTM processing.

## 4. Model architecture

The initial neural architecture is a Bidirectional Long Short-Term Memory network.

### Architecture configuration

| Component | Configuration |
| --- | --- |
| Embedding layer | 100 dimensions |
| LSTM hidden size | 128 |
| LSTM direction | Bidirectional |
| Combined representation | 256 dimensions |
| Dropout | 0.3 |
| Output layer | 6 units |
| Activation for probabilities | Sigmoid |

### Architecture flow

```text
Input Comment
      |
      v
Custom Tokenizer
      |
      v
Vocabulary Encoding
      |
      v
Embedding Layer (100)
      |
      v
Bidirectional LSTM
Hidden Size = 128
      |
      v
Concatenate Forward
and Backward States
      |
      v
Dropout (0.3)
      |
      v
Linear Layer (6 outputs)
      |
      v
Logits
      |
      v
Sigmoid (Inference)
      |
      v
Six Independent
Class Probabilities
```

The model uses `pack_padded_sequence` to process variable-length sequences without treating padded tokens as meaningful input.

The final forward and backward hidden states are concatenated and passed to a six-unit linear classification layer.

The model produces six logits. Sigmoid is applied during inference to obtain probabilities.

## 5. Training configuration

The initial model was trained using the following configuration:

| Parameter | Value |
| --- | --- |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 1 |
| Dropout | 0.3 |
| Positive class weighting | Disabled |

No `pos_weight` was supplied to the loss function in this experiment.

### Hardware acceleration

The development environment was migrated from CPU-only PyTorch to CUDA-enabled PyTorch.

| Component | Configuration |
| --- | --- |
| PyTorch | 2.5.1 |
| CUDA runtime reported by PyTorch | 12.4 |
| GPU | NVIDIA GeForce RTX 2050 |

The training pipeline uses device-agnostic tensor placement, allowing execution on CUDA when available and CPU otherwise.

### Recorded training observations

The notebook contains multiple recorded runs. They are kept separate below because they do not have identical loss values.

| Recorded run | Device | Epochs | Training loss | Validation loss | Time |
| --- | --- | ---: | ---: | ---: | --- |
| CPU benchmark | CPU | 1 | 0.076618 | 0.056350 | ~12 min |
| GPU benchmark note | RTX 2050 | 1 | 0.076628 | 0.057901 | 48 sec |
| Notebook training cell | RTX 2050 | 1 | 0.076322 | 0.056624 | Not recorded in that cell |
| Subsequent GPU run | RTX 2050 | 1 | 0.079302 | 0.056304 | 58 sec |

These observations demonstrate that GPU acceleration substantially reduces training time. However, they are separate runs and should not be interpreted as a controlled speed benchmark with identical initial weights and execution conditions.

## 6. Evaluation methodology

The model was evaluated using the validation dataset.

The inference pipeline collects ground-truth labels and model probabilities. The sigmoid function converts logits into probabilities.

A default classification threshold of 0.5 was initially applied:

```python
y_pred = (y_prob >= 0.5).int()
```

The evaluation implementation calculates:

- Per-label precision.
- Per-label recall.
- Per-label F1-score.
- Micro precision and recall.
- Micro F1-score.
- Macro F1-score.
- Weighted F1-score.
- Individual 2 × 2 confusion matrices for all six labels.

### Metric definitions

**Precision:** The proportion of predicted positive examples that are actually positive.

**Recall:** The proportion of actual positive examples correctly identified.

**F1-score:** The harmonic mean of precision and recall.

**Micro F1:** Aggregates decisions across all labels before calculating the score.

**Macro F1:** Calculates F1 independently for each label and takes the unweighted average.

**Weighted F1:** Calculates the average of per-label F1 scores weighted by each label's support.

Macro F1 is particularly important for this project because it gives rare labels equal importance in the final average.

## 7. Baseline evaluation results

### 7.1 Per-label performance

At the default threshold of 0.5:

| Label | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Toxic | 0.8595 | 0.6443 | 0.7365 | 3,056 |
| Severe toxic | 0.6081 | 0.1402 | 0.2278 | 321 |
| Obscene | 0.8719 | 0.6391 | 0.7376 | 1,715 |
| Threat | 0.0000 | 0.0000 | 0.0000 | 74 |
| Insult | 0.8031 | 0.5130 | 0.6261 | 1,614 |
| Identity hate | 0.0000 | 0.0000 | 0.0000 | 294 |

### 7.2 Aggregate performance

| Metric | Score |
| --- | ---: |
| Micro precision | 0.8463 |
| Micro recall | 0.5567 |
| Micro F1 | 0.6716 |
| Macro F1 | 0.3880 |
| Weighted F1 | 0.6502 |

### 7.3 Confusion matrices

The confusion matrix format is:

```text
[[True Negative, False Positive],
 [False Negative, True Positive]]
```

**Toxic**

```text
[[28537   322]
 [ 1087  1969]]
```

**Severe toxic**

```text
[[31565    29]
 [  276    45]]
```

**Obscene**

```text
[[30039   161]
 [  619  1096]]
```

**Threat**

```text
[[31841     0]
 [   74     0]]
```

**Insult**

```text
[[30098   203]
 [  786   828]]
```

**Identity hate**

```text
[[31621     0]
 [  294     0]]
```

The model did not predict any positive examples for `threat` or `identity_hate` at the default threshold.

## 8. Probability distribution analysis

To investigate the poor performance of minority labels, predicted probabilities were examined separately for actual positive examples.

| Label | Mean probability | Median probability | Maximum probability |
| --- | ---: | ---: | ---: |
| Toxic | 0.6290 | 0.7467 | 0.9987 |
| Severe toxic | 0.2572 | 0.2343 | 0.7768 |
| Obscene | 0.6239 | 0.7634 | 0.9963 |
| Threat | 0.0321 | 0.0280 | 0.0896 |
| Insult | 0.4940 | 0.5170 | 0.9639 |
| Identity hate | 0.0618 | 0.0555 | 0.3084 |

### Observations

**Common labels**

The model assigns relatively high probabilities to positive examples of `toxic`, `obscene`, and `insult`.

This suggests that the model has learned useful patterns for these categories.

**Severe toxic**

Positive examples receive a wide range of probabilities. Some examples receive relatively high probabilities, but many remain below the default threshold.

**Threat**

Even the highest probability assigned to an actual positive example is only 0.0896.

This indicates that the model struggles to identify this category, rather than simply making a small threshold-calibration error.

**Identity hate**

The model assigns some positive examples probabilities above 0.1, but the overall distribution remains low.

This suggests that lowering the threshold can recover some positive predictions.

## 9. Decision threshold experiments

The default threshold of 0.5 was investigated to understand its effect on individual label performance.

The following thresholds were evaluated:

`0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50`

### Per-label F1 results

| Label | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toxic | 0.5415 | 0.6449 | 0.6950 | 0.7204 | 0.7465 | 0.7465 | 0.7365 |
| Severe toxic | 0.3920 | 0.4449 | 0.4585 | 0.4759 | 0.4291 | 0.3690 | 0.2278 |
| Obscene | 0.6033 | 0.6948 | 0.7438 | 0.7635 | 0.7694 | 0.7567 | 0.7376 |
| Threat | 0.0962 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Insult | 0.5477 | 0.6352 | 0.6699 | 0.6872 | 0.6884 | 0.6620 | 0.6261 |
| Identity hate | 0.2197 | 0.1860 | 0.0741 | 0.0199 | 0.0068 | 0.0000 | 0.0000 |

### Best observed thresholds in this sweep

| Label | Best tested threshold | Best observed F1 |
| --- | ---: | ---: |
| Toxic | 0.30 or 0.40 | 0.7465 |
| Severe toxic | 0.20 | 0.4759 |
| Obscene | 0.30 | 0.7694 |
| Threat | 0.05 | 0.0962 |
| Insult | 0.30 | 0.6884 |
| Identity hate | 0.05 | 0.2197 |

These results demonstrate that a single threshold of 0.5 is not necessarily suitable for every label.

The lower thresholds improve F1 for several minority labels, but they can also increase false-positive predictions.

These are exploratory validation results. The selected thresholds must not be treated as final unbiased test performance because they were selected using the validation set.

## 10. Comparison with the classical ML baseline

The initial classical baseline uses TF-IDF features and One-vs-Rest Logistic Regression.

| Metric | TF-IDF + Logistic Regression | BiLSTM baseline |
| --- | ---: | ---: |
| Micro F1 | 0.6746 | 0.6716 |
| Macro F1 | 0.4793 | 0.3880 |
| Weighted F1 | 0.6656 | 0.6502 |

### Interpretation

The BiLSTM achieves a micro F1 close to the classical baseline.

However, the classical model performs better in macro F1 and weighted F1.

The difference in macro F1 indicates that the initial BiLSTM is less effective across the complete set of labels, particularly the minority categories.

At this stage, the neural architecture has not demonstrated an overall improvement over the classical approach.

## 11. Problems identified

The first experiment revealed several areas that need improvement.

### 11.1 Severe class imbalance

The positive examples are distributed unevenly across the six labels.

The rare categories have substantially fewer positive examples than the common categories.

### 11.2 Weak minority-label predictions

The model predicts no positive examples for `threat` and `identity_hate` at the default threshold.

### 11.3 Low macro F1

The macro F1 of 0.3880 shows that the model's performance is inconsistent across labels.

### 11.4 Threshold sensitivity

The F1-scores change considerably when the decision threshold is modified.

This demonstrates that threshold selection is an important part of multi-label classification.

### 11.5 Limited training duration

The initial baseline was trained for only one epoch.

Further training experiments are required to investigate whether the model can improve with additional optimization.

## 12. Conclusions

Experiment 01 successfully established an end-to-end BiLSTM training and evaluation pipeline for multi-label toxic comment classification.

The implementation includes custom tokenization, vocabulary construction, dynamic padding, packed sequence processing, GPU training, prediction generation, and comprehensive multi-label evaluation.

The initial model performs reasonably well on common categories such as `toxic` and `obscene`, but struggles with rare categories such as `threat` and `identity_hate`.

The threshold experiments show that some of this performance loss can be recovered by changing decision thresholds. However, the probability distribution analysis suggests that threshold adjustment alone is insufficient for the `threat` category.

The classical TF-IDF baseline currently performs better in macro F1.

Therefore, the next experiment will investigate whether positive class weighting can improve the model's ability to learn minority categories.

## 13. Next experiment

**Experiment 02 — BiLSTM with Positive Class Weighting**

The next experiment will introduce the `pos_weight` parameter in `BCEWithLogitsLoss`.

The initial plan is to:

1. Calculate positive class weights using the training split only.
2. Cap extreme weights at 20 for the first controlled experiment.
3. Reinitialize the BiLSTM model and optimizer.
4. Train for five epochs.
5. Monitor training loss, validation loss, micro F1, and macro F1.
6. Inspect per-label precision, recall, and F1.
7. Repeat the probability distribution and threshold experiments.

The primary objective is to improve minority-label performance without significantly degrading performance on common categories.

---

**Experiment artifacts**

- Notebook: `notebooks/model_training_ex1.ipynb`
- Model implementation: `src/deep_learning/model.py`
- Dataset implementation: `src/deep_learning/dataset.py`
- Training implementation: `src/deep_learning/trainer.py`
- Evaluation implementation: `src/evaluation/metrics.py`
- Data split: `data/processed/split_indices.npz`
- Recorded model checkpoint: `models/bilstm_epoch_1.pt`
