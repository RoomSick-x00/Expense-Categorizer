from rules import apply_rules

def predict_expense(text, amount, ml_model):
    rule_prediction = apply_rules(text, amount)

    if rule_prediction:
        return rule_prediction

    return ml_model.predict([text])[0]
