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

Pure machine learning systems also fail when:
* Text input is extremely short
* Context is missing
* Statistical signals are weak

This project explores whether **classical machine learning**, combined with minimal domain rules, can handle these ambiguities better than either approach alone.

---

## 2. Approach

Expense categorization is treated as a **supervised text classification** problem with a hybrid decision pipeline.

### Model Pipeline

* Text input → TF-IDF vectorization
* Vectorized text → Classical ML classifier
* Optional rule-based override before ML prediction

### Models Used

* **v0.1**: Multinomial Naive Bayes (baseline)
* **v0.2**: Logistic Regression (comparison model)
* **v0.3**: Logistic Regression + rule-based preprocessor

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

---

### v0.2 — Logistic Regression

* Accuracy: ~90%+
* Clear improvement over Naive Bayes
* Errors are fewer and more reasonable

Typical failure cases:
* Very short descriptions
* Brand-only inputs
* Low-information text

---

### v0.3 — Hybrid Rules + ML

v0.3 introduces a **rule-first, ML-second** architecture.

Decision flow:

Input → Rule Engine → ML Model (fallback)


#### Rule Engine (v0.3)

Rules are applied only when ML is known to be unreliable.

**Merchant-based rules**
* Known merchants are mapped directly to categories
* Example: "swiggy dinner" → Food

**Ambiguity detection**
* Descriptions with fewer than 3 words are considered ambiguous
* Example: "dinner", "coffee", "lunch"

**Amount-based rule (current behavior)**
* For ambiguous descriptions:
  * Low-amount expenses default to Food
  * Higher amounts fall through to ML

This behavior is documented as a known limitation in v0.3.

---

### Typical Failure Modes (v0.3)

* Ambiguous descriptions with higher amounts may still be misclassified
* ML predictions on weak text remain imperfect
* No confidence score is exposed

These errors are acceptable given the scope and constraints of this version.

---

## 5. Design Decisions

### Why Hybrid Rules + ML

* ML struggles on weak textual signals
* Rules encode strong domain knowledge cheaply
* Hybrid systems reduce obvious misclassifications
* Behavior remains explainable and deterministic

---

### Why No Deep Learning

* Dataset too small
* Reduced interpretability
* Unnecessary complexity
* Violates engineering simplicity

---

### Why No Overengineering

* No hyperparameter tuning
* No feature explosion
* No premature optimization

Each version adds **one dimension of complexity only**.

---

## Current Status

* v0.1: Baseline model complete
* v0.2: Model comparison complete
* v0.3: Hybrid rule + ML system implemented

Next step: feature engineering and refined rule precedence (v0.4).
