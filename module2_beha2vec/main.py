import pickle
import pandas as pd
import os

# STEP 1: load sequences
with open("../module1/clean_sequences.pkl", "rb") as f:
    sequences = pickle.load(f)

rows = []

# STEP 2: convert to CSV
for user_id, seq in enumerate(sequences):
    for i, item in enumerate(seq):
        rows.append({
            "user_id": user_id,
            "timestamp": i,
            "pageview_URL": item
        })

df = pd.DataFrame(rows)

os.makedirs("input", exist_ok=True)
df.to_csv("input/data.csv", index=False)

print("✅ Data ready")

# STEP 3: train
os.system("python -m scripts.model_training")

# STEP 4: embeddings
os.system("python -m scripts.embedding_generation")

print("✅ ALL DONE 🎉")