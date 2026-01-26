import joblib #a fast and reliable way to save and load python objects to disk
from rules import apply_rules

# Load trained model ONCE
ml_model = joblib.load("model.pkl")

def predict_expense(text, amount):
    rule_prediction = apply_rules(text, amount)

    if rule_prediction:
        return rule_prediction

    return ml_model.predict([text])[0]


if __name__ == "__main__":
    text = input("Enter expense description: ")
    amount = float(input("Enter amount: "))

    category = predict_expense(text, amount)
    print("Predicted category:", category)
