from transformers import pipeline
import numpy as np
import random
import re

LABELS = [
    "Clothing",
    "Electronics",
    "Bags",
    "Drinkware",
    "Office",
    "Kids",
    "Accessories",
    "Unknown"
]

WORD_TO_CATEGORY = {
    "shirt": "Clothing",
    "tshirt": "Clothing",
    "hoodie": "Clothing",
    "jacket": "Clothing",
    "backpack": "Bags",
    "shopping": "Bags",
    "laptop": "Electronics",
    "charger": "Electronics",
    "speaker": "Electronics",
    "bottle": "Drinkware",
    "steel": "Drinkware",
    "kids": "Kids",
    "baby": "Kids",
    "notebook": "Office",
    "journal": "Office",
    "sunglasses": "Accessories",
    "rfid": "Accessories",
    "bag": "Bags",
    "pack": "Bags",
    "cup": "Drinkware",
    "pen": "Office",
    "mouse": "Electronics",
}

BRAND_WORDS = {
    "google", "android", "youtube", "mens", "womens", "men", "women",
    "unisex", "kids", "youth", "adult", "new", "old", "best", "top",
    "limited", "edition", "size", "color", "black", "white", "red",
    "blue", "green", "grey", "gray", "navy", "pink", "bttf", "moonshot",
    "redesign", "apparel", "product", "item", "shop", "store", "buy",
    "sale", "deal", "offer", "s", "xs", "xl", "xxl", "sm", "md", "lg",
    "the", "a", "an", "and", "or", "of", "in", "for", "with", "performance",
    "dry", "full", "zip", "fit", "slim", "regular", "classic", "premium",
    "pro", "plus", "max", "mini", "lite", "ultra", "super", "v", "vneck",
    "neck", "crew", "zip", "snap", "button", "sleeve", "short", "long",
}

PRODUCT_TYPES = {
    "shirt", "tee", "tshirt", "hoodie", "jacket", "wear", "top", "polo",
    "sweater", "sweatshirt", "vest", "coat", "pants", "shorts", "leggings",
    "dress", "skirt", "blouse", "cardigan", "pullover", "fleece", "jersey",
    "bag", "backpack", "tote", "pouch", "wallet", "purse", "sling", "duffel",
    "laptop", "charger", "speaker", "mouse", "keyboard", "cable", "adapter",
    "headphone", "earphone", "phone", "tablet", "device", "usb", "hub",
    "bottle", "mug", "cup", "tumbler", "flask", "thermos", "jug",
    "notebook", "journal", "pen", "pencil", "pad", "planner", "binder",
    "sunglasses", "watch", "bracelet", "keychain", "lanyard", "badge",
    "toy", "game", "puzzle", "book", "kit",
}


def extract_product_type(slug):
    words = [w for w in slug.split("-") if w and len(w) > 1]
    filtered = [w for w in words if w not in BRAND_WORDS]
    if not filtered:
        filtered = words
    for word in reversed(filtered):
        if word in PRODUCT_TYPES:
            return word
    meaningful = [w for w in filtered if len(w) > 2]
    if meaningful:
        return "-".join(meaningful[-2:]) if len(meaningful) >= 2 else meaningful[-1]
    return "unknown-product"


def clean_url(url):
    return url.lower().replace("-", " ").replace("/", " ")


def extract_slug(url):
    slug = url.split("/")[-1].split("?")[0]
    slug = slug.lower()
    slug = slug.replace("+", "-")
    slug = slug.replace(".axd", "")
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-").strip()
    if not slug:
        return "unknown-product"
    product_type = extract_product_type(slug)
    return product_type


def normalize_token(token):
    if "::" not in token:
        token = f"unknown::{token}"
    parts = token.lower().split("::", 1)
    category = parts[0].strip().replace(" ", "-")
    slug = parts[1].strip().replace(" ", "-") if len(parts) > 1 else "unknown-product"
    slug = slug.replace("+", "-")
    slug = slug.replace(".axd", "")
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-").strip()
    if not slug:
        slug = "unknown-product"
    product_type = extract_product_type(slug)
    if product_type == category:
        product_type = "unknown-product"
    return f"{category}::{product_type}"


def smart_rule_classify(url):
    cleaned = clean_url(url)
    for word, category in WORD_TO_CATEGORY.items():
        if word in cleaned:
            return category
    return None


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

cache = {}


def classify_url(url):
    cleaned = clean_url(url)
    if cleaned in cache:
        return cache[cleaned]
    if not cleaned or not cleaned.strip():
        raise ValueError("URL empty hai!")
    result = classifier(cleaned, LABELS)
    label = result["labels"][0]
    confidence = result["scores"][0]
    if confidence < 0.4:
        return "Unknown"
    cache[cleaned] = label
    return label


def transform_sequences(sequences):
    if not sequences:
        print("⚠️ Empty input!")
        return []
    results = []
    for seq in sequences:
        new_seq = []
        for url in seq:
            try:
                slug = extract_slug(url)
                label = smart_rule_classify(url)
                if label:
                    token = normalize_token(f"{label.lower()}::{slug}")
                    new_seq.append(token)
                    continue
                label = classify_url(url)
                token = normalize_token(f"{label.lower()}::{slug}")
                new_seq.append(token)
            except Exception as e:
                print(f"Error: {e}")
                new_seq.append("unknown::unknown-product")
        results.append(new_seq)
    return results


global_token_to_id = {}
global_id_to_token = {}


def build_global_mapping(sequences):
    """Build deterministic global token↔id mapping (sorted for stability)"""
    all_tokens = sorted(set(token for seq in sequences for token in seq))
    global_token_to_id.clear()
    global_id_to_token.clear()
    for idx, token in enumerate(all_tokens):
        global_token_to_id[token] = idx
        global_id_to_token[idx] = token
    print(f"Global vocab size: {len(global_token_to_id)}")


def encode_sequences(sequences):
    encoded_data = []
    for seq in sequences:
        encoded_seq = []
        for token in seq:
            if token in global_token_to_id:
                encoded_seq.append(global_token_to_id[token])
            else:
                new_id = len(global_token_to_id)
                global_token_to_id[token] = new_id
                global_id_to_token[new_id] = token
                encoded_seq.append(new_id)
        encoded_data.append(encoded_seq)
    return encoded_data


def decode_sequences(encoded_sequences):
    decoded_data = []
    for seq in encoded_sequences:
        decoded_seq = []
        for idx in seq:
            token = global_id_to_token.get(idx, "unknown::unknown-product")
            decoded_seq.append(token)
        decoded_data.append(decoded_seq)
    return decoded_data


def sincerity_filter(sequence, keep_ratio=0.2):
    arr = np.array(sequence)
    fft_vals = np.fft.fft(arr)
    n = len(fft_vals)
    cutoff = int(n * keep_ratio)
    fft_filtered = np.copy(fft_vals)
    fft_filtered[cutoff: n - cutoff] = 0
    cleaned = np.fft.ifft(fft_filtered)
    cleaned = np.real(cleaned)
    cleaned = np.round(cleaned).astype(int)
    max_id = max(global_id_to_token.keys()) if global_id_to_token else 0
    cleaned = np.clip(cleaned, 0, max_id)
    return cleaned.tolist()


# ✅ FIX WARNING 1: rotate fallback instead of always accessories
FALLBACK_CATEGORIES = ["accessories", "office", "electronics", "bags", "drinkware"]

SIMILAR_CATEGORIES = {
    "Clothing": ["Accessories", "Bags"],
    "Electronics": ["Accessories"],
    "Bags": ["Accessories"],
    "Drinkware": ["Office"],
    "Office": ["Accessories"],
    "Kids": ["Clothing"],
    "Accessories": ["Clothing"]
}


def semantic_insertion(sequence, min_length=5):
    new_seq = sequence.copy()
    while len(new_seq) < min_length:
        last = new_seq[-1]
        last_category = last.split("::")[0].capitalize() if "::" in last else last
        if last_category in SIMILAR_CATEGORIES:
            new_category = random.choice(SIMILAR_CATEGORIES[last_category])
            new_item = f"{new_category.lower()}::unknown-product"
        else:
            new_item = last
        new_seq.append(new_item)
    return new_seq


# ✅ FIX WARNING 1: rotate across FALLBACK_CATEGORIES
def semantic_substitution(sequence):
    new_seq = []
    for i, item in enumerate(sequence):
        if item.startswith("unknown::"):
            fallback = FALLBACK_CATEGORIES[i % len(FALLBACK_CATEGORIES)]
            new_seq.append(f"{fallback}::unknown-product")
        else:
            new_seq.append(item)
    return new_seq


# ✅ FIX WARNING 2: min_length 5 → 3 to reduce synthetic padding
def cold_start_fix(sequence):
    seq = semantic_substitution(sequence)
    seq = semantic_insertion(seq, min_length=3)
    return seq


def remove_consecutive_duplicates(seq):
    """Remove consecutive duplicates: ['a','a','b'] → ['a','b']"""
    if not seq:
        return seq
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result
# from transformers import pipeline

# # Categories (Unknown add kar diya safe side ke liye)
# LABELS = [
#     "Clothing",
#     "Electronics", 
#     "Bags",
#     "Drinkware",
#     "Office",
#     "Kids",
#     "Accessories",
#     "Unknown"
# ]
# WORD_TO_CATEGORY = {
#     "shirt": "Clothing",
#     "tshirt": "Clothing",
#     "hoodie": "Clothing",
#     "jacket": "Clothing",
    
#     "backpack": "Bags",
#     "shopping": "Bags",
    
#     "laptop": "Electronics",
#     "charger": "Electronics",
#     "speaker": "Electronics",
    
#     "bottle": "Drinkware",
#     "steel": "Drinkware",
    
#     "kids": "Kids",
#     "baby": "Kids",
    
#     "notebook": "Office",
#     "journal": "Office",
    
#     "sunglasses": "Accessories",
#     "rfid": "Accessories",
#     "bag": "Bags",
# "pack": "Bags",
# "bottle": "Drinkware",
# "cup": "Drinkware",
# "pen": "Office",
# "mouse": "Electronics",
# }
# def clean_url(url):
#     return url.lower().replace("-", " ").replace("/", " ")

# def smart_rule_classify(url):
#     url = clean_url(url)
   
    
#     for word, category in WORD_TO_CATEGORY.items():
#         if word in url:
#             return category
    
#     return None
# # Model load (sirf ek baar)
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",framework="pt")

# # Cache (same URL dobara process na ho)
# cache = {}


# def classify_url(url):
#     # Cache check
#     url = clean_url(url)
#     if url in cache:
        
#         return cache[url]
    
#     # Empty URL check
#     if not url or not url.strip():
#         raise ValueError("URL empty hai!")
    
    
    
#     result = classifier(url, LABELS)
    
#     label = result["labels"][0]
#     confidence = result["scores"][0]
    
    
#     if confidence < 0.4:
#         return "Unknown"
#     # Save in cache
#     cache[url] = label
    
#     return label


# def transform_sequences(sequences):
#     if not sequences:
#         print("⚠️ Empty input!")
#         return []
    
#     results = []
    
#     for seq in sequences:
#         new_seq = []
        
#         for url in seq:
#             try:
#                 # STEP 1: Data-driven rules
#                 label = smart_rule_classify(url)
#                 if label:
#                     new_seq.append(label)
#                     continue
#                 label = classify_url(url)
#                 new_seq.append(label)
#             except Exception as e:
                
#                 print(f"Error: {e}")
#                 new_seq.append("Unknown")
#         results.append(new_seq)
    
#     return results

# #Encoding
# label_to_id = {}

# for i, label in enumerate(LABELS):
#     label_to_id[label] = i

# print("Mapping:", label_to_id)
# def encode_sequences(sequences):
#     encoded_data = []
    
#     for seq in sequences:
#         encoded_seq = []
        
#         for label in seq:
#             # agar label nahi mila toh Unknown use karo
#             if label in label_to_id:
#                 encoded_seq.append(label_to_id[label])
#             else:
#                 encoded_seq.append(label_to_id["Unknown"])
        
#         encoded_data.append(encoded_seq)
    
#     return encoded_data


   



# import numpy as np

# def sincerity_filter(sequence, keep_ratio=0.2):
#     """
#     FFT-based low-pass filter to remove high-frequency noise
    
#     Args:
#         sequence: encoded sequence (list of ints)
#         keep_ratio: kitna low-frequency retain karna hai (0.2–0.5 best)
    
#     Returns:
#         cleaned sequence (list of ints)
#     """
    
#     # STEP 1: convert to numpy
#     arr = np.array(sequence)
    
#     # STEP 2: FFT
#     fft_vals = np.fft.fft(arr)
    
#     n = len(fft_vals)
#     cutoff = int(n * keep_ratio)
    
#     # STEP 3: Low-pass filter (IMPORTANT 🔥)
#     fft_filtered = np.copy(fft_vals)
    
#     # high-frequency remove (middle part zero)
#     fft_filtered[cutoff : n - cutoff] = 0
    
#     # STEP 4: Inverse FFT
#     cleaned = np.fft.ifft(fft_filtered)
    
#     # STEP 5: real values lo + round karo
#     cleaned = np.real(cleaned)
#     cleaned = np.round(cleaned).astype(int)
    
#     return cleaned.tolist()
# seq = [3, 7, 1, 6, 2]

# filtered_seq = sincerity_filter(seq, keep_ratio=0.4)

# print("Original:", seq)
# print("Filtered:", filtered_seq)


# SIMILAR_CATEGORIES = {
#     "Clothing": ["Accessories", "Bags"],
#     "Electronics": ["Accessories"],
#     "Bags": ["Accessories"],
#     "Drinkware": ["Office"],
#     "Office": ["Accessories"],
#     "Kids": ["Clothing"],
#     "Accessories": ["Clothing"]
# }
# import random

# def semantic_insertion(sequence, min_length=5):
    
#     new_seq = sequence.copy()
    
#     while len(new_seq) < min_length:
#         last = new_seq[-1]
        
#         if last in SIMILAR_CATEGORIES:
#             new_item = random.choice(SIMILAR_CATEGORIES[last])
#         else:
#             new_item = last
        
#         new_seq.append(new_item)
    
#     return new_seq

# def semantic_substitution(sequence):
    
#     new_seq = []
    
#     for item in sequence:
#         if item == "Unknown":
#             # replace with safe fallback
#             new_seq.append("Accessories")
#         else:
#             new_seq.append(item)
    
#     return new_seq

# def cold_start_fix(sequence):
    
#     # STEP 1: substitute Unknown
#     seq = semantic_substitution(sequence)
    
#     # STEP 2: insert if too short
#     seq = semantic_insertion(seq, min_length=5)
    
#     return seq

# seq = ["Clothing",'Accessories',"Electronics",  "Bags"]

# fixed_seq = cold_start_fix(seq)

# print("Fixed:", fixed_seq)