# IMDB Sentiment Analysis with LSTM & GRU + GloVe Embeddings

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Model Description](#-model-description)
  - [LSTMClassifier](#lstmclassifier)
  - [GRUClassifier](#gruclassifier)
- [Requirements](#-requirements)
- [Workflow](#-workflow)
- [Performance Comparison](#-performance-comparison)

---

## Project Overview

This project tackles a **binary sentiment classification** task: given an IMDB movie review, the model predicts whether the sentiment is **positive** (1) or **negative** (0).

The pipeline is built from scratch using **Pure PyTorch** with no high-level training frameworks. Key design decisions include:

- Pretrained **GloVe** word embeddings (50d, 200d, 300d) as the input representation layer, with fine-tuning enabled during training.
- A recurrent encoder — either **LSTM** or **GRU** — to capture sequential context from the review text.
- Text preprocessing via **NLTK** (tokenization, stopword removal, lowercasing).
- Models are evaluated using **Precision**, **Recall**, and **F1-score** on a held-out test set.

---

## Dataset Description

| Property | Details |
|---|---|
| **Source** | [IMDB Dataset (GitHub)](https://github.com/SK7here/Movie-Review-Sentiment-Analysis/raw/refs/heads/master/IMDB-Dataset.csv) |
| **Size** | 50,000 reviews |
| **Columns** | `review` (raw text), `sentiment` (`positive` / `negative`) |
| **Language** | English |
| **Task** | Binary Classification |

**Sample data:**

| review | sentiment |
|---|---|
| One of the other reviewers has mentioned that... | positive |
| A wonderful little production. The filming... | positive |
| I thought this was a wonderful way to spend... | negative |

**Preprocessing steps:**
- Split text at delimiter `\n\n---\n\n`, keeping only the main content.
- Replace hyphens with spaces, remove special characters via regex.
- Lowercase all text, remove digits and English stopwords (NLTK).
- Label encoding: `positive → 1`, `negative → 0`.
- Data split: **60% Train / 20% Validation / 20% Test** (stratified by label, `random_state=42`).

**Pretrained Embeddings:**

| GloVe File | Dimensions | Trained On |
|---|---|---|
| `glove.6B.50d.txt` | 50 | 6 billion tokens |
| `glove.6B.200d.txt` | 200 | 6 billion tokens |
| `glove.6B.300d.txt` | 300 | 6 billion tokens |

Special tokens: `<PAD>` (zero vector) and `<UNK>` (random normal vector) are prepended to the vocabulary.

---

## Model Description

### LSTMClassifier

The baseline model encodes tokenized input using a single-layer **LSTM**, then passes the final hidden state through a linear classification head.

```
Input (token indices)  →  Shape: (batch_size, seq_len)
        │
Embedding Layer (GloVe pretrained, fine-tuned)
        │  Shape: (batch_size, seq_len, embedding_dim)
        │
LSTM Encoder (hidden_dim = 128, batch_first=True)
        │  Take h_n[-1]  →  Shape: (batch_size, hidden_dim)
        │
Linear(hidden_dim, num_classes=2)
        │
Logits  →  Shape: (batch_size, 2)
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding | GloVe 6B (50d / 200d / 300d) |
| Max sequence length | 128 |
| Batch size | 32 |
| Hidden dim | 128 |
| Num classes | 2 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss function | CrossEntropyLoss |
| Epochs | 10 |
| Fine-tune embeddings | Yes |

---

### GRUClassifier

A drop-in replacement for the LSTM encoder, using a single-layer **GRU** instead. The architecture is otherwise identical — same embedding layer, same FC head — making it a clean controlled comparison.

```
Input (token indices)  →  Shape: (batch_size, seq_len)
        │
Embedding Layer (GloVe pretrained, fine-tuned)
        │  Shape: (batch_size, seq_len, embedding_dim)
        │
GRU Encoder (hidden_dim = 128, batch_first=True)
        │  Take h_n[-1]  →  Shape: (batch_size, hidden_dim)
        │
Linear(hidden_dim, num_classes=2)
        │
Logits  →  Shape: (batch_size, 2)
```

**Hyperparameters:** Same as LSTMClassifier above, with `nn.GRU` replacing `nn.LSTM`.

---

## Requirements

```bash
pip install torch scikit-learn pandas numpy tqdm nltk
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch>=2.0.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.65.0
nltk>=3.8.0
```

**Download GloVe embeddings:**
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

**Download NLTK data (run once in Python):**
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
```

> **Note:** A CUDA-compatible GPU is strongly recommended for training. The code automatically selects `cuda` if available, otherwise falls back to `cpu`.

---

## Workflow

```
1. Environment Setup
   └── Install dependencies, download GloVe & NLTK data

2. Data Download & Loading
   └── Download IMDB-Dataset.csv
   └── Load with pandas, extract `review` and `sentiment`

3. Data Preprocessing
   └── Clean text: remove special chars, lowercase, strip stopwords & digits
   └── Label encoding: positive → 1, negative → 0
   └── Train / Val / Test split: 60% / 20% / 20% (stratified)

4. GloVe Embeddings
   └── Load glove.6B.Xd.txt → build word2idx + embedding_matrix
   └── Prepend <PAD> (zero vector) and <UNK> (random vector)

5. Dataset & DataLoader
   └── Wrap data in IMDBDataset (PyTorch Dataset)
   └── Tokenize with NLTK word_tokenize, map to indices
   └── Pad / truncate to max_len = 128
   └── Create DataLoader (batch_size = 32)

6. Model Initialization
   └── Build LSTMClassifier or GRUClassifier
   └── Load pretrained GloVe weights into Embedding layer
   └── Enable fine-tuning (requires_grad = True)
   └── Move model to DEVICE

7. Training Loop (10 Epochs)
   └── Forward pass → CrossEntropyLoss → Backprop → Adam update
   └── Evaluate on validation set each epoch (Loss, P, R, F1)
   └── Save best model checkpoint (highest Val F1)

8. Evaluation on Test Set
   └── Load best checkpoint
   └── Compute Precision, Recall, F1-score on test set

9. Model Comparison
   └── LSTM vs GRU across GloVe 50d / 200d / 300d
```

---

## Performance Comparison

Results on the **test set** after 10 epochs of training:

| Model | GloVe | Test Loss | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| **LSTMClassifier** | 200d | 0.3135 | 0.8799 | 0.8676 | 0.8737 |
| **LSTMClassifier** | 300d | 0.2969 | 0.8836 | 0.8760 | 0.8798 |
| **GRUClassifier** | 300d | **0.2813** | 0.8543 | **0.9206** | **0.8862** |

### Conclusion

- **GRUClassifier with GloVe 300d** achieves the best overall performance, outperforming LSTM on Test Loss, Recall, and F1-score. GRU's simpler gating mechanism (reset gate + update gate vs. LSTM's input/forget/output gates) appears to generalize better on this task.

- **LSTMClassifier** achieves higher **Precision** (0.8836 vs 0.8543), meaning it makes fewer false positive predictions when classifying a review as positive. This can be preferable in applications where precision is critical.

- **GRUClassifier** achieves significantly higher **Recall** (0.9206 vs 0.8760), meaning it captures more true positive sentiment cases and misses fewer positive reviews — beneficial when false negatives are costly.

- Increasing the GloVe embedding dimension from 50d → 200d → 300d consistently improves performance, as richer word representations provide the recurrent encoder with more semantic signal.

- **Conclusion:** For this binary sentiment task, GRU with GloVe 300d provides the best balance of Precision and Recall. If prioritizing Precision, LSTM remains a competitive choice. In either case, using high-dimensional pretrained embeddings (300d) is recommended over lower-dimensional alternatives.

---
