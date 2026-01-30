def text_len_feature(text: str) -> int:
    return len(text.split())

def amount_bucket(amount: float)-> str:
    if amount < 70:
        return "low"
    if amount <= 600:
        return "medium"
    else:
        return "high"
    
from rules import MERCHANT_RULES

def has_known_merchant(text: str) -> int:
    text = text.lower()
    for merchant in MERCHANT_RULES:
        if merchant in text:
            return 1    
    return 0
    