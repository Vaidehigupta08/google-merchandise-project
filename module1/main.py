import pickle
from collections import Counter
from module1 import (
    transform_sequences,
    encode_sequences,
    decode_sequences,
    build_global_mapping,
    sincerity_filter,
    cold_start_fix,
    normalize_token,
    remove_consecutive_duplicates,
)

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
with open("final_sequences.pkl", "rb") as f:
    data = pickle.load(f)

print("\n🔥 RAW DATA SAMPLE:")
print(data[:2])

# -------------------------------
# STEP 2: UNIQUE URL EXTRACTION
# -------------------------------

# FIX ISSUE 3: Filter noise URLs before processing
NOISE_KEYWORDS = ["home", "index", "landing", "default"]

def is_noise_url(url):
    url_lower = url.lower()
    return any(keyword in url_lower for keyword in NOISE_KEYWORDS)

unique_urls = list(set(
    url for seq in data for url in seq
    if not is_noise_url(url)
))

print(f"Total URLs: {sum(len(seq) for seq in data)}")
print(f"Unique URLs (after noise filter): {len(unique_urls)}")

# -------------------------------
# STEP 3: CLASSIFY ONLY UNIQUE URLs
# -------------------------------
classified_unique = transform_sequences([unique_urls])[0]
url_map = dict(zip(unique_urls, classified_unique))

classified = []
for seq in data:
    # skip noise URLs when building classified sequences
    mapped = [url_map[url] for url in seq if url in url_map]
    classified.append(mapped)

print("\n🔥 Classification Done!")
print(classified[:2])

flat = [item for seq in classified for item in seq]
print("\n🔥 CATEGORY DISTRIBUTION:")
print(Counter(item.split("::")[0] for item in flat))

# -------------------------------
# STEP 4: Clean sequences
# -------------------------------
clean_sequences = []

for seq in classified:

    # NORMALIZE tokens
    seq = [normalize_token(x) for x in seq]

    # FIX ISSUE 2: Replace unknown tokens → accessories::unknown-product (do NOT remove)
    seq = [
        "accessories::unknown-product" if x.startswith("unknown::") else x
        for x in seq
    ]

    # FIX ISSUE 4: Clothing dominance at 60% threshold (was 70%)
    clothing_count = sum(1 for x in seq if x.startswith("clothing::"))
    if seq and clothing_count > len(seq) * 0.6:
        seq = [x for x in seq if not x.startswith("clothing::")]

    # FIX ISSUE 1: Guard against empty sequence before cold_start_fix
    if len(seq) == 0:
        continue  # skip empty — nothing to fix

    # SHORT SEQUENCE HANDLING: fix first, then skip if still too short
    if len(seq) < 2:
        seq = cold_start_fix(seq)  # safe: seq has at least 1 element

    if len(seq) < 2:
        continue

    # DUPLICATE HANDLING: remove consecutive duplicates only
    seq = remove_consecutive_duplicates(seq)

    # EMPTY SEQUENCES: skip
    if not seq:
        continue

    # COLD START FIX (skip second call if already long enough)
    fixed = seq if len(seq) >= 2 else cold_start_fix(seq)

    # Normalize again after cold_start_fix
    fixed = [normalize_token(x) for x in fixed]

    # Replace unknown again after normalization
    fixed = [
        "accessories::unknown-product" if x.startswith("unknown::") else x
        for x in fixed
    ]

    # Final empty check
    if not fixed:
        continue

    clean_sequences.append(fixed)

# -------------------------------
# STEP 5: Build global mapping on classified (more token coverage)
# -------------------------------
build_global_mapping(classified)

# -------------------------------
# STEP 6: Encode → Sincerity Filter → Decode
# -------------------------------
encoded = encode_sequences(clean_sequences)

filtered_encoded = []
for seq in encoded:
    filtered = sincerity_filter(seq, keep_ratio=0.4)
    filtered_encoded.append(filtered)

clean_sequences = decode_sequences(filtered_encoded)

# -------------------------------
# POST-DECODE VALIDATION
# -------------------------------
validated_sequences = []

for seq in clean_sequences:
    seq = [str(x) for x in seq]
    seq = [normalize_token(x) for x in seq]

    # Replace unknown after decode as well
    seq = [
        "accessories::unknown-product" if x.startswith("unknown::") else x
        for x in seq
    ]

    seq = remove_consecutive_duplicates(seq)

    if not seq:
        continue

    valid = all("::" in token for token in seq)
    if not valid:
        continue

    validated_sequences.append(seq)

clean_sequences = validated_sequences

# -------------------------------
# STEP 7: Save
# -------------------------------
with open("clean_sequences.pkl", "wb") as f:
    pickle.dump(clean_sequences, f)

print("\n✅ Clean sequences saved!")

flat_clean = [x for seq in clean_sequences for x in seq]
print("\n🔥 CLEAN DATA DISTRIBUTION:")
print(Counter(item.split("::")[0] for item in flat_clean))

print("Original sequences:", len(classified))
print("Clean sequences:", len(clean_sequences))

print(clean_sequences[:2])
print(type(clean_sequences[0][0]) if clean_sequences and clean_sequences[0] else "No data!")

# -------------------------------
# FINAL VALIDATION ASSERTIONS
# -------------------------------
assert all(isinstance(seq, list) for seq in clean_sequences), "❌ Non-list sequence!"
assert all(len(seq) > 0 for seq in clean_sequences), "❌ Empty sequence found!"
assert all(isinstance(t, str) for seq in clean_sequences for t in seq), "❌ Non-string token!"
assert all("::" in t for seq in clean_sequences for t in seq), "❌ Token missing '::' separator!"
print("\n✅ ALL VALIDATIONS PASSED!")











# import pickle

# with open("final_sequences.pkl", "rb") as f:
#     data = pickle.load(f)

# from module1 import (
#     transform_sequences,
#     encode_sequences,
#     sincerity_filter,
#     cold_start_fix
# )

# # # 🔥 TEST DATA (auto generate)
# # test_data = [
# #     ["https://shop.com/shirt", "https://shop.com/laptop"],
# #     ["https://shop.com/backpack", "https://shop.com/product/xyz123"],
# #     ["https://shop.com/hoodie"],
# #     ["https://shop.com/random-item-999"]
# # ]

# # print("\n🔥 STEP 1: RAW DATA")
# # print(test_data)

# print("\n🔥 RAW DATA SAMPLE:")
# print(data[:2])

# # -------------------------------
# # STEP 2: Classification
# # -------------------------------
# # 🔥 STEP 2: UNIQUE URL EXTRACTION
# unique_urls = list(set(url for seq in data for url in seq))

# print(f"Total URLs: {sum(len(seq) for seq in data)}")
# print(f"Unique URLs: {len(unique_urls)}")

# # 🔥 STEP 3: CLASSIFY ONLY UNIQUE URLs
# classified_unique = transform_sequences([unique_urls])[0]

# # mapping bana lo
# url_map = dict(zip(unique_urls, classified_unique))

# # original sequences pe mapping apply karo
# classified = []
# for seq in data:
#     classified.append([url_map[url] for url in seq])

# print("\n🔥 Classification Done!")
# print(classified[:2])
# from collections import Counter

# flat = [item for seq in classified for item in seq]
# print("\n🔥 CATEGORY DISTRIBUTION:")
# print(Counter(flat))

# # -------------------------------
# # STEP 3: Encoding
# # -------------------------------
# # encoded = encode_sequences(classified)

# # print("\n🔥 STEP 3: ENCODED (Categories → Numbers)")
# # print(encoded)

# # # -------------------------------
# # # STEP 4: FFT Sincerity Filter
# # # -------------------------------
# # filtered_sequences = []

# # print("\n🔥 STEP 4: SINCERITY FILTER (FFT)")

# # for seq in encoded:
# #     filtered = sincerity_filter(seq)
# #     filtered_sequences.append(filtered)
    
# #     print(f"Original: {seq}")
# #     print(f"Filtered: {filtered}")
# #     print("------")

# # # -------------------------------
# # # STEP 5: Cold Start Fix
# # # -------------------------------
# # print("\n🔥 STEP 5: COLD START FIX")

# # for seq in classified:
# #     fixed = cold_start_fix(seq)
    
# #     print(f"Original: {seq}")
# #     print(f"Fixed: {fixed}")
# #     print("------")

# clean_sequences = []

# for seq in classified:
    
#     # 🔥 STEP 1: remove Unknown
#     seq = [x for x in seq if x != "Unknown"]
    
#     # 🔥 STEP 2: Clothing dominance reduce
#     if seq.count("Clothing") > len(seq) * 0.7:
#         seq = [x for x in seq if x != "Clothing"]
    
#     # 🔥 STEP 3: skip weak sequences
#     if len(seq) < 2:
#         continue
    
#     # 🔥 STEP 4: cold start fix
#     fixed = cold_start_fix(seq)
    
#     clean_sequences.append(fixed)
    
# with open("clean_sequences.pkl", "wb") as f:
#     pickle.dump(clean_sequences, f)

# print("\n✅ Clean sequences saved!")

# from collections import Counter

# flat_clean = [x for seq in clean_sequences for x in seq]

# print("\n🔥 CLEAN DATA DISTRIBUTION:")
# print(Counter(flat_clean))
# print("Original sequences:", len(classified))
# print("Clean sequences:", len(clean_sequences))
# #print("\n✅ PIPELINE COMPLETE 🚀")