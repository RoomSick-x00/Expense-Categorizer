import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:\\Users\\mailp\\Desktop\\ExpP\\Expense-Categorizer\\expenses.csv")

X = df["text"]
y = df["category"]

# Train-test split (engineering discipline)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": y_pred
})

# Filter wrong predictions
errors = results[results["actual"] != results["predicted"]]

print(errors.head(10))