MERCHANT_RULES = {
    "swiggy": "Food",
    "zomato": "Food",
    "ubereats": "Food",
    "uber": "Transport",
    "ola": "Transport",
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "apollo": "Health",
    "pharmeasy": "Health"
}

FOOD_AMOUNT_THRESHOLD = 100

def normalize_text(text: str) -> str:
    return text.lower().strip()

def merchant_rule(text: str):
    text = normalize_text(text)

    for merchant, category in MERCHANT_RULES.items():
        if merchant in text:
            return category
        
    return None

def is_ambiguous(text: str) -> bool:
    words = text.split()
    return len(words) < 3

def amount_rule(text: str, amount: float):
    if not is_ambiguous(text):
        return None

    if amount < FOOD_AMOUNT_THRESHOLD:
        return "Food"

    return None


def apply_rules(text: str, amount: float):
    category = merchant_rule(text)
    if category:
        return category

    category = amount_rule(text, amount)
    if category:
        return category

    return None

