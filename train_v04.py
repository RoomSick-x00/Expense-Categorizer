import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from features import (
    bucket_amount,
    extract_structured_features
)

df = pd.read_csv("expenses.csv")

structured_df = df["text"].apply(extract_structured_features)
structured_df = pd.DataFrame(structured_df.tolist())

df = pd.concat([df, structured_df], axis=1)
df["amount_bucket"] = df["amount"].apply(bucket_amount)

df.columns = df.columns.str.strip()

X = df[
    ["text",
     "amount_bucket",
     "text_length",
     "has_known_merchant",
     "has_food_word",
     "has_health_word",
     "has_entertainment_word"]
]

y = df["category"].str.lower().str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

text_transformer = TfidfVectorizer()

numeric_features = ["text_length", "has_known_merchant",
                    "has_food_word", "has_health_word",
                    "has_entertainment_word"]

categorical_features = ["amount_bucket"]

preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(), "text"),
        ("cat", OneHotEncoder(), ["amount_bucket"]),
        ("num", "passthrough", [
            "text_length",
            "has_known_merchant",
            "has_food_word",
            "has_health_word",
            "has_entertainment_word"
        ]),
    ]
)

print(X.head())
print(X.columns)


model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        solver="liblinear",
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("v0.4 Accuracy:", accuracy)

