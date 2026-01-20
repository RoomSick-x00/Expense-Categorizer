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

## Current Status

* v0.1: Baseline model complete
* v0.2: Model comparison complete
* Logistic Regression selected as default

Next step: feature engineering and hybrid rule + ML systems (v0.3).
