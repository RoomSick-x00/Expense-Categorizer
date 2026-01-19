# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# # Load dataset
# df = pd.read_csv("C:\\Users\\mailp\\Desktop\\ExpP\\Expense-Categorizer\\expenses.csv")

# X = df["text"]
# y = df["category"].str.strip().str.lower()

# # Train-test split (engineering discipline)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # ML pipeline
# model = Pipeline([
#     ("vectorizer", TfidfVectorizer()),
#     ("classifier", MultinomialNB())
# ])

# # Train
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)

# results = pd.DataFrame({
#     "text": X_test.reset_index(drop=True),
#     "actual": y_test.reset_index(drop=True),
#     "predicted": y_pred
# })

# # Filter wrong predictions
# errors = results[results["actual"] != results["predicted"]]

# print(errors.head(10))
# print(len(errors), "errors out of", len(X_test))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:\\Users\\mailp\\Desktop\\ExpP\\Expense-Categorizer\\expenses.csv")

X = df["text"]
y = df["category"].str.strip().str.lower()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_accuracy)

lr_model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        solver="liblinear",
        max_iter=1000
    ))
])

lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

lr_results = pd.DataFrame({
    "text": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": lr_pred
})

lr_errors = lr_results[lr_results["actual"] != lr_results["predicted"]]

print(lr_errors.head(10))
print(len(lr_errors), "errors out of", len(X_test))


