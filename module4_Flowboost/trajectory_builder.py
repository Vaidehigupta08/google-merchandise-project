"""
trajectory_builder.py
=====================
Module 4 — Step 1

Converts outputs from Module 2 + 3 into CFM-ready trajectory dataset.

Input:
    - module2_beha2vec/embeddings.json        (user_id -> 128d vector)
    - module3_ELCRec/outputs/cluster_map.json (user_id -> cluster_id)
    - module3_ELCRec/outputs/intents.json     (cluster_id -> intent_vector)
    - module2_beha2vec/input/data.csv         (user_id, timestamp, pageview_URL)

Output:
    - module4_CFM/outputs/trajectories.json
      {
        "user_id": "42",
        "cluster_id": 2,
        "x1": [128 floats],          # normalized full embedding (target state)
        "condition": [128 floats],   # intent vector of cluster (CFM condition)
        "url_sequence": ["Bags", "Clothing", ...]
      }
"""

import json
import numpy as np
import pandas as pd
import os
from collections import defaultdict


# ==========================
# Paths
# ==========================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_PATH     = os.path.join(BASE, "module2_beha2vec", "embeddings.json")
CLUSTER_PATH = os.path.join(BASE, "module3_ELCRec", "outputs", "cluster_map.json")
INTENT_PATH  = os.path.join(BASE, "module3_ELCRec", "outputs", "intents.json")
DATA_PATH    = os.path.join(BASE, "module2_beha2vec", "input", "data.csv")
OUT_PATH     = os.path.join(BASE, "module4_Flowboost", "outputs", "trajectories.json")


def normalize(vec):
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / (norm + 1e-8)).tolist()


def build_trajectories():
    print("📂 Loading data...")

    with open(EMB_PATH) as f:
        embeddings = json.load(f)

    with open(CLUSTER_PATH) as f:
        cluster_map = json.load(f)

    with open(INTENT_PATH) as f:
        intents = json.load(f)

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["user_id", "timestamp"])

    # Build url sequence per user
    url_sequences = (
        df.groupby("user_id")["pageview_URL"]
        .apply(list)
        .to_dict()
    )

    trajectories = []
    skipped = 0

    for uid_str, emb_vec in embeddings.items():
        uid_int = int(uid_str)

        # Get cluster
        cluster_id = cluster_map.get(uid_str)
        if cluster_id is None:
            skipped += 1
            continue

        # Get intent vector for this cluster
        intent_data = intents.get(str(cluster_id))
        if intent_data is None or len(intent_data["intent_vector"]) == 0:
            skipped += 1
            continue

        # Normalize embedding -> x1 (target state)
        x1 = normalize(emb_vec)

        # Condition = cluster intent vector (already normalized)
        condition = intent_data["intent_vector"]

        # URL sequence
        url_seq = url_sequences.get(uid_int, [])

        trajectories.append({
            "user_id":      uid_str,
            "cluster_id":   int(cluster_id),
            "x1":           x1,
            "condition":    condition,
            "url_sequence": url_seq
        })

    print(f"✅ Built {len(trajectories)} trajectories  ({skipped} skipped)")

    # Cluster distribution check
    from collections import Counter
    dist = Counter(t["cluster_id"] for t in trajectories)
    print(f"   Cluster distribution: {dict(sorted(dist.items()))}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"💾 Saved → {OUT_PATH}")
    return trajectories


if __name__ == "__main__":
    build_trajectories()
