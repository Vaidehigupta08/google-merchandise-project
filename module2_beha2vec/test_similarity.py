import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load embeddings
with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

# convert to arrays
user_ids = list(embeddings.keys())
vectors = np.array(list(embeddings.values()))

# pick one user
target_idx = 0
target_vector = vectors[target_idx].reshape(1, -1)

# similarity
similarities = cosine_similarity(target_vector, vectors)[0]

# 🔥 sort all users (descending)
sorted_indices = np.argsort(similarities)[::-1]

# 🔥 remove self + take top 5
top_k = []
for idx in sorted_indices:
    if idx != target_idx:
        top_k.append(idx)
    if len(top_k) == 5:
        break

# output
print(f"\n🎯 Target User: {user_ids[target_idx]}")
print("🔥 Top Similar Users:")

for idx in top_k:
    print(f"User {user_ids[idx]} → Score: {similarities[idx]:.4f}")