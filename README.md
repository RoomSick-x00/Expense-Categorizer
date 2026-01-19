# Expense Categorizer (v0.1)

## 1. Problem Statement

Personal expense descriptions are short, informal, and inconsistent:

* "swiggy dinner"
* "uber ride"
* "amazon purchase"

Rule-based categorization (if–else, keyword matching) fails quickly because:

* Wording varies widely for the same intent
* New brands or phrases break existing rules
* Rules become hard to scale and maintain
* Ambiguous descriptions cannot be handled cleanly

This project explores whether **classical machine learning** can learn patterns from real examples instead of relying on brittle rules.

---

## 2. Approach

This project uses a **classical supervised machine learning pipeline**.

### Learning Paradigm

* **Supervised learning**
* Input: expense description (text)
* Output: one fixed expense category

### Text Representation

* **TF-IDF (Term Frequency – Inverse Document Frequency)**
* Converts raw text into numeric feature vectors
* Emphasizes rare but informative words (e.g., "swiggy", "uber")
* De-emphasizes common words (e.g., "to", "from", "the")

### Classifier

* **Multinomial Naive Bayes**
* Well-suited for text classification
* Fast, interpretable, and effective for small datasets
* Serves as a strong and widely-used baseline

The final pipeline:

```
Raw Text → TF-IDF Vectorizer → Naive Bayes Classifier → Category
```

---

## 3. Dataset

### Dataset Characteristics

* **Manually curated dataset**
* Approximately **~N samples**
* Each row contains:

  * Expense description (text)
  * Expense category (label)

### Fixed Categories (v0.1)

* Food
* Transport
* Shopping
* Bills
* Entertainment
* Health

Categories are **fixed and mutually exclusive** for v0.1.

### Label Normalization

Before training, category labels are normalized:

* Trimmed whitespace
* Converted to lowercase

This prevents silent bugs caused by visually identical but unequal labels (e.g., `"Shopping"` vs `"Shopping "`).

---

## 4. Results

### Accuracy

* Accuracy varies depending on the train–test split
* Typical accuracy observed: **~65–75%**

This is expected given:

* Small dataset size
* Short and ambiguous text descriptions

### Typical Failure Modes

* **Ambiguous descriptions**

  * e.g., "amazon purchase", "payment"

* **Category overlap**

  * e.g., "movie rental" confused with transport-related rentals

* **Sparse categories**

  * Categories with fewer examples (e.g., Health) perform worse

* **Very short text inputs**

  * Fewer words → weaker TF-IDF signal

### Why These Errors Are Acceptable

* Errors are explainable and consistent
* Failures reflect data limitations, not random behavior
* Model behavior matches expectations of TF-IDF–based systems

The goal of v0.1 is **understanding and correctness**, not maximum accuracy.

---

## 5. Design Decisions

### Why No Deep Learning

* Deep learning requires significantly more data
* Adds unnecessary complexity for this problem size
* Hides core learning intuition behind abstractions

Classical ML provides better transparency and learning value.

### Why No Overengineering

* No online learning
* No auto-correction loops
* No hyperparameter tuning obsession
* No model ensembling

Each design choice keeps the system:

* Understandable
* Debuggable
* Easy to reason about end-to-end

---

## Version Status

* **v0.1 is frozen**
* Focus: correctness, intuition, and ML fundamentals
* Future versions will build incrementally on this foundation
