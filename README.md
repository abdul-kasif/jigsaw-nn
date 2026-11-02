# Deep Learning for Multi-Label Toxic Comment Classification

## 1. Project Overview

This project aims to develop a deep learning-based Natural Language Processing (NLP) system for detecting different types of toxic content in online comments. The objective is to classify each comment according to six possible toxicity categories: **toxic, severe toxic, obscene, threat, insult, and identity hate**.

The project will treat the problem as a **multi-label text classification task**, because a single comment can belong to multiple toxicity categories simultaneously. For example, one comment may be classified as both toxic and obscene.

The main goal is to investigate how deep learning models can learn meaningful representations of natural language and identify multiple forms of toxic content from raw textual input.

---

## 2. Project Setup

### i. Prerequisites

- Micromamba (**v2.10+**)

1. **Install micromamba**
   - For Linux, macOS, and Git Bash on Windows, install with:

     ```bash
     "\${SHELL}" <(curl -L micro.mamba.pm/install.sh)
     ```

   - For Windows PowerShell, install with:

     ```powershell
     Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1).Content)
     ```

2. **Check micromamba installation**

   ```bash
   micromamba --version
   ```

### ii. Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/abdul-kasif/jigsaw-nn.git
   cd jigsaw-nn/
   ```

2. **Create micromamba environment and install dependencies**

   ```bash
   micromamba env create -f environment.yml
   ```

3. **Activate micromamba environment**

   ```bash
   micromamba activate ml_env
   ```

---

## 3. Dataset Selection

The dataset selected for this project is the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle.

The dataset originates from a completed Kaggle competition focused on identifying toxic behavior in online comments. The training dataset contains **159,571 comments**, with each comment annotated using six toxicity labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The dataset contains a `comment_text` field containing the original text and six corresponding binary target labels.

### Dataset Characteristics

| Property                |          Value |
| ----------------------- | -------------: |
| Training samples        |        159,571 |
| Text field              | `comment_text` |
| Number of target labels |              6 |
| Classification type     |    Multi-label |
| Missing comments        |              0 |
| Multi-label comments    |          9,865 |
| Average comment length  |    67.27 words |
| Median comment length   |       36 words |
| Maximum comment length  |    1,411 words |
| Class imbalance         |        Present |

The dataset is substantially larger than the minimum requirement of 10,000 documents for an NLP project.

---

## 4. Why This Dataset Was Selected

The Jigsaw dataset was selected because it provides a sufficiently large and challenging NLP problem where deep learning can be meaningfully applied.

### 4.1 Large-scale Text Dataset

The dataset contains **159,571 textual documents**, which is significantly above the required 10,000-document minimum. This provides enough data to train and evaluate neural NLP models.

### 4.2 Multi-Label Classification

Unlike a simple single-class classification problem, the dataset contains six independent toxicity categories. A comment can have more than one positive label.

There are **9,865 comments with multiple toxicity labels**, demonstrating that the categories are not mutually exclusive.

This makes the task more complex than conventional single-label classification.

### 4.3 Natural Language Structure

The input consists of real-world user-generated text rather than structured numerical data. The comments contain different vocabulary, sentence structures, word combinations, punctuation, spelling variations, and contextual information.

Consequently, the text needs to be transformed into numerical representations through techniques such as **tokenization, vocabulary mapping, embeddings, or subword tokenization** before it can be processed by a neural network.

### 4.4 Class Imbalance

The dataset contains substantial differences in the frequency of the toxicity categories. For example, the `toxic` category contains 15,294 positive examples, whereas `threat` contains only 478.

This imbalance creates an additional machine-learning challenge and requires careful evaluation beyond overall accuracy.

### 4.5 Suitable for Comparing Classical and Deep Learning Approaches

The dataset also allows a meaningful comparison between a traditional NLP approach and a deep learning approach.

A classical baseline can use:

```text
TF-IDF → Logistic Regression
```

while the deep learning approach can learn representations directly from tokenized text.

This comparison will help evaluate whether the additional complexity of deep learning provides benefits for this NLP task.

---

## 5. Planned Deep Learning Approach

The project will investigate **deep learning for multi-label text classification**.

The initial architecture will follow this general pipeline:

```text
Raw Comment
     │
     ▼
Text Preprocessing
     │
     ▼
Tokenization
     │
     ▼
Token IDs / Vocabulary Mapping
     │
     ▼
Embedding Layer
     │
     ▼
Sequential Neural Network
     │
     ▼
Dense Layer
     │
     ▼
Six Sigmoid Outputs
     │
     ├── Toxic
     ├── Severe Toxic
     ├── Obscene
     ├── Threat
     ├── Insult
     └── Identity Hate
```

### 5.1 Text Preprocessing

The raw comments will first be examined and prepared for model training. The preprocessing stage may include handling unnecessary whitespace, normalizing text where appropriate, and preparing the text for tokenization.

The preprocessing will be designed carefully because excessive cleaning can remove information that may be useful for toxicity detection.

### 5.2 Tokenization

The comments will be converted into sequences of tokens.

Depending on the selected model, the project may investigate either:

- word-level tokenization with a learned vocabulary, or
- subword tokenization.

Subword tokenization is particularly useful for handling uncommon words, spelling variations, and previously unseen words.

### 5.3 Embeddings

The tokenized text will be transformed into dense numerical representations using an embedding layer.

The embedding representation will allow the neural network to learn relationships between words and their surrounding linguistic context during training.

### 5.4 Deep Learning Model

The primary model will initially be based on a **Bidirectional LSTM (BiLSTM)** or **GRU** architecture.

A recurrent neural network is suitable for this task because comments are sequences of words, and the meaning of a word can depend on the surrounding context.

A possible architecture is:

```text
Token IDs
    ↓
Embedding Layer
    ↓
Bidirectional LSTM
    ↓
Dropout
    ↓
Dense Layer
    ↓
Six Sigmoid Outputs
```

A BiLSTM can process information from both directions of a sequence, allowing the model to use contextual information from preceding and following words.

---

## 6. Multi-Label Prediction

The final layer will contain **six independent output neurons**, one for each toxicity category.

A sigmoid activation will be used for each output because each category is an independent binary prediction.

For example:

```text
Comment
   ↓
Neural Network
   ↓
[0.92, 0.03, 0.81, 0.01, 0.76, 0.04]
```

These values represent the model's predicted probability for:

```text
toxic          → 0.92
severe_toxic   → 0.03
obscene        → 0.81
threat         → 0.01
insult         → 0.76
identity_hate  → 0.04
```

A classification threshold will then be used to convert these probabilities into the final multi-label predictions.

---

## 7. Classical Baseline

Before evaluating the deep learning model, a traditional machine-learning baseline will be established.

The baseline will use:

```text
Raw Text
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Multi-Label Predictions
```

This baseline provides a reference point for determining whether the deep learning model is able to capture useful linguistic patterns beyond sparse TF-IDF representations.

---

## 8. Evaluation

Because the dataset is highly imbalanced and multi-label, accuracy alone will not be sufficient.

The models will be evaluated using metrics such as:

- Precision
- Recall
- F1-score
- Macro F1-score
- Micro F1-score
- Per-label performance

Particular attention will be given to the rare classes, especially `threat` and `identity_hate`, because their substantially smaller number of positive examples can make them difficult to classify.

A confusion matrix or per-label classification report will also be used to analyze model behavior.

---

## 9. Expected Outcome

The expected outcome of the project is a trained deep learning model capable of identifying multiple forms of toxicity within an online comment.

The project will also provide an experimental comparison between a traditional TF-IDF-based approach and a neural NLP approach.

The main research question will be:

> **How effectively can a deep learning model learn contextual representations of online comments for multi-label toxicity classification compared with a traditional TF-IDF-based machine-learning approach?**

The project will focus on understanding the complete NLP pipeline, from raw text processing and tokenization to neural representation learning, multi-label prediction, and evaluation.
