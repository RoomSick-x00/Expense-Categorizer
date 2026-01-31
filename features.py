def text_len_feature(text: str) -> int:
    return len(text.split())

def bucket_amount(amount: int) -> str:
    if amount < 100:
        return "low"
    elif amount < 500:
        return "medium"
    elif amount < 2000:
        return "high"
    else:
        return "very_high"

from rules import MERCHANT_RULES

def has_known_merchant(text: str) -> int:
    text = text.lower()
    for merchant in MERCHANT_RULES:
        if merchant in text:
            return 1    
    return 0
    
FOOD_WORDS = ["dinner", "lunch", "meal", "snack", "coffee"]
HEALTH_WORDS = ["doctor", "medicine", "hospital", "clinic"]
ENTERTAINMENT_WORDS = ["movie", "netflix", "game"]

def contains_any(text: str, words: list[str])-> int:
    text= text.lower()
    return int(any(word in text for word in words))

def keyword_features(text: str) -> dict:
    return {
        "has_food_word": contains_any(text, FOOD_WORDS),
        "has_health_word": contains_any(text, HEALTH_WORDS),
        "has_entertainment_word": contains_any(text, ENTERTAINMENT_WORDS),
    }
    
def extract_structured_features(text: str, amount: float) -> dict:
    features = {
        "text_length": text_len_feature(text),
        "amount_bucket": bucket_amount(amount),
        "has_known_merchant": has_known_merchant(text),
    }

    features.update(keyword_features(text))
    return features

