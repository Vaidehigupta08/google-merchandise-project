import json
import os
import pandas as pd
from functools import lru_cache

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(BASE, *parts)


# ── loaders ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_embeddings() -> dict:
    path = _path("module2_beha2vec", "embeddings.json")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_cluster_map() -> dict:
    path = _path("module3_ELCRec", "outputs", "cluster_map.json")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_intents() -> dict:
    path = _path("module3_ELCRec", "outputs", "intents.json")
    with open(path) as f:
        return json.load(f)


# ✅ FIXED: predictions loader
@lru_cache(maxsize=1)
def load_predictions() -> list:
    path = _path("module4_Flowboost", "outputs", "predictions.json")

    print("📂 Loading predictions from:", path)

    if not os.path.exists(path):
        print("❌ predictions.json NOT FOUND")
        return []

    with open(path) as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} predictions")
    return data


@lru_cache(maxsize=1)
def load_clickstream() -> pd.DataFrame:
    path = _path("module2_beha2vec", "input", "data.csv")
    return pd.read_csv(path).sort_values(["user_id", "timestamp"])


# ── derived helpers ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_url_vocab() -> list:
    df = load_clickstream()
    return sorted(df["pageview_URL"].unique().tolist())


@lru_cache(maxsize=1)
def get_cluster_url_stats() -> dict:
    from collections import Counter, defaultdict

    df = load_clickstream()
    cmap = load_cluster_map()
    stats = defaultdict(Counter)

    for _, row in df.iterrows():
        cid = cmap.get(str(row["user_id"]))
        if cid is not None:
            stats[int(cid)][row["pageview_URL"]] += 1

    return dict(stats)


# ✅ FIXED: user_id normalization (VERY IMPORTANT 🔥)
@lru_cache(maxsize=1)
def get_predictions_by_user() -> dict:
    data = load_predictions()

    result = {}
    for p in data:
        uid = str(p["user_id"])  # force string
        result[uid] = p

    print("👥 Available user_ids:", list(result.keys())[:10])
    return result


@lru_cache(maxsize=1)
def get_cluster_sizes() -> dict:
    intents = load_intents()
    return {int(k): v["size"] for k, v in intents.items()}


def invalidate_cache():
    load_embeddings.cache_clear()
    load_cluster_map.cache_clear()
    load_intents.cache_clear()
    load_predictions.cache_clear()
    load_clickstream.cache_clear()
    get_url_vocab.cache_clear()
    get_cluster_url_stats.cache_clear()
    get_predictions_by_user.cache_clear()
    get_cluster_sizes.cache_clear()


# """
# data_loader.py
# ==============
# Module 5 — Centralized Data Loader

# Loads all outputs from Module 2, 3, 4 into memory once.
# All other files import from here — single source of truth.
# """

# import json
# import os
# import pandas as pd
# from functools import lru_cache

# BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# def _path(*parts):
#     return os.path.join(BASE, *parts)


# # ── loaders ────────────────────────────────────────────────────────────

# @lru_cache(maxsize=1)
# def load_embeddings() -> dict:
#     path = _path("module2_beha2vec", "embeddings.json")
#     with open(path) as f:
#         return json.load(f)


# @lru_cache(maxsize=1)
# def load_cluster_map() -> dict:
#     path = _path("module3_ELCRec", "outputs", "cluster_map.json")
#     with open(path) as f:
#         return json.load(f)


# @lru_cache(maxsize=1)
# def load_intents() -> dict:
#     path = _path("module3_ELCRec", "outputs", "intents.json")
#     with open(path) as f:
#         return json.load(f)


# @lru_cache(maxsize=1)
# def load_predictions() -> list:
#     path = _path("module4_CFM", "outputs", "predictions.json")
#     if not os.path.exists(path):
#         return []
#     with open(path) as f:
#         return json.load(f)


# @lru_cache(maxsize=1)
# def load_clickstream() -> pd.DataFrame:
#     path = _path("module2_beha2vec", "input", "data.csv")
#     return pd.read_csv(path).sort_values(["user_id", "timestamp"])


# # ── derived helpers ────────────────────────────────────────────────────

# @lru_cache(maxsize=1)
# def get_url_vocab() -> list:
#     df = load_clickstream()
#     return sorted(df["pageview_URL"].unique().tolist())


# @lru_cache(maxsize=1)
# def get_cluster_url_stats() -> dict:
#     """Returns {cluster_id: {url: count}} for all clusters."""
#     from collections import Counter, defaultdict
#     df        = load_clickstream()
#     cmap      = load_cluster_map()
#     stats     = defaultdict(Counter)
#     for _, row in df.iterrows():
#         cid = cmap.get(str(row["user_id"]))
#         if cid is not None:
#             stats[int(cid)][row["pageview_URL"]] += 1
#     return dict(stats)


# @lru_cache(maxsize=1)
# def get_predictions_by_user() -> dict:
#     """Returns {user_id: prediction_dict}."""
#     return {p["user_id"]: p for p in load_predictions()}


# @lru_cache(maxsize=1)
# def get_cluster_sizes() -> dict:
#     """Returns {cluster_id: user_count}."""
#     intents = load_intents()
#     return {int(k): v["size"] for k, v in intents.items()}


# def invalidate_cache():
#     """Call this if any source file is updated at runtime."""
#     load_embeddings.cache_clear()
#     load_cluster_map.cache_clear()
#     load_intents.cache_clear()
#     load_predictions.cache_clear()
#     load_clickstream.cache_clear()
#     get_url_vocab.cache_clear()
#     get_cluster_url_stats.cache_clear()
#     get_predictions_by_user.cache_clear()
#     get_cluster_sizes.cache_clear()
