# Expense Categorizer

A classical machine learning project to automatically categorize expense descriptions into fixed categories using text-based features.

This project is intentionally built **without deep learning**, focusing on fundamentals, intuition, and engineering discipline.

---

## 1. Problem Statement

Expense categorization is often implemented using rule-based systems:

* Keyword matching ("pizza" → Food)
* Merchant hardcoding ("Uber" → Transport)

These approaches fail because:

* Language is ambiguous ("movie rental" vs "car rental")
* Descriptions are short and noisy
* Rules do not generalize
* Maintenance cost grows exponentially

This project explores whether **classical machine learning** can handle these ambiguities better than rigid rules.

---

## 2. Approach

We treat expense categorization as a **supervised text classification** problem.

### Model Pipeline

* Text input → TF-IDF vectorization
* Vectorized text → Classical ML classifier

### Models Used

* **v0.1**: Multinomial Naive Bayes (baseline)
* **v0.2**: Logistic Regression (comparison model)

Why classical ML:

* Interpretable behavior
* Works well on small datasets
* Fast training and inference
* Easy to debug

Deep learning is intentionally avoided.

---

## 3. Dataset

* Manually curated expense descriptions
* ~N labeled samples
* Short, real-world text entries

### Categories (fixed)

* Food
* Transport
* Shopping
* Entertainment
* Health
* Utilities

### Label Normalization

* Lowercasing
* Consistent category naming
* Removal of accidental duplicates

Dataset is kept small on purpose to expose model limitations.

---

## 4. Results

### v0.1 — Naive Bayes

* Accuracy: ~65–70%
* Strengths:

  * Fast
  * Stable
* Weaknesses:

  * Confusion between similar categories
  * Sensitive to word overlap

### v0.2 — Logistic Regression

* Accuracy: ~90%+
* Clear improvement over Naive Bayes
* Errors are fewer and more reasonable

### Typical Failure Modes

* Very short descriptions ("Xbox Series X")
* Brand-only inputs
* Overlapping semantic categories (Food vs Health)

These errors are acceptable given the lack of context and metadata.

---

## 5. Design Decisions

### Why No Deep Learning

* Dataset too small
* Harder to interpret
* Adds unnecessary complexity
* Violates engineering simplicity

### Why No Overengineering

* No hyperparameter tuning
* No feature explosion
* No premature optimization

Each version adds **one dimension of complexity only**.

---

## 6. v0.3 — Hybrid Rules + ML

v0.3 introduces a **rule-first, ML-second** architecture to handle cases where pure ML is unreliable.

### Decision Flow

Input → Rule Engine → ML Model (fallback)

#### Rule Engine (v0.3)

* **Merchant-based rules**

  * Known merchants are mapped directly to categories
  * Example: `"swiggy dinner" → Food`

* **Ambiguity detection**

  * Descriptions with fewer than 3 words are treated as ambiguous
  * Examples: `"dinner"`, `"coffee"`, `"lunch"`

* **Amount-based heuristic**

  * For ambiguous descriptions:

    * Low-amount expenses default to Food
    * Higher amounts fall through to ML

This behavior is documented as a known limitation.

### Typical Failure Modes (v0.3)

* Ambiguous text with higher amounts may still be misclassified
* Weak textual signals remain difficult
* No confidence score is exposed

These errors are acceptable given the constraints.

---

## 7. v0.4 — Structured Feature Engineering

v0.4 focuses on improving robustness using **lightweight structured features** while staying fully classical and interpretable.

The goal is **not** to maximize accuracy, but to understand trade-offs.

### Added Features

**Text-derived features**

* `text_length`: number of words
* `has_known_merchant`
* `has_food_words`
* `has_health_words`
* `has_entertainment_words`

**Amount-derived feature**

* `amount_bucket`: discretized transaction amount (`low`, `medium`, `high`, `very_high`)

All features are simple, binary or categorical, and explainable.

---

### Model Pipeline (v0.4)

* TF-IDF on raw text
* One-hot encoding for `amount_bucket`
* Numeric/boolean features passed directly
* Logistic Regression classifier

All preprocessing is handled inside a single `ColumnTransformer` + `Pipeline`.

No hyperparameter tuning is performed.

---

### Results (v0.4)

* **Accuracy ≈ 79%**

Accuracy decreased compared to v0.3.

This is **expected**.

---

### Why Accuracy Fell

* Dataset is small and imbalanced
* Rare categories have very few samples
* Feature space increased
* Model must balance sparse TF-IDF with dense features

Accuracy alone is not a reliable metric at this scale.

---

### What v0.4 Demonstrates

* Feature engineering does not guarantee better accuracy
* More features ≠ better performance
* Data quality dominates model choice
* Structured features improve interpretability, not miracles

---

## 8. Design Philosophy

* No deep learning
* No feature explosion
* No premature optimization
* One new complexity dimension per version

Understanding behavior is prioritized over chasing metrics.

---

## Current Status

* v0.1: Naive Bayes baseline
* v0.2: Logistic Regression comparison
* v0.3: Hybrid rules + ML
* **v0.4: Structured feature engineering**

Next steps will be taken only with a clear engineering reason.

thinkinhg of adding new prod