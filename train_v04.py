import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def bucket_amount(amount: int) -> str:
    if amount < 100:
        return "low"
    elif amount < 500:
        return "medium"
    elif amount < 2000:
        return "high"
    else:
        return "very_high"


df = pd.read_csv("expenses.csv")

df.columns = df.columns.str.strip()

df["amount_bucket"] = df["amount"].apply(bucket_amount)

X = df[["text", "amount"]]
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
        ("cat", OneHotEncoder(), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

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

