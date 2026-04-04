"""
feedback_loop.py — Module 5 (FIXED)

Fixes:
1. min_feedback_events = 10 (100 se kam — realistic threshold)
2. compute_reward_weights() auto-called on every log_feedback
3. get_cluster_reward_stats() mein acceptance_rate properly shown
"""

import json
import os
from datetime import datetime, timezone
from collections import defaultdict

from module5_agent.m5_data_loader import load_cluster_map, get_predictions_by_user, load_intents


BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH     = os.path.join(BASE, "module5_agent", "outputs", "reward_log.json")
WEIGHTS_PATH = os.path.join(BASE, "module5_agent", "outputs", "reward_weights.json")


REWARD_MAP = {
    "accepted": 1.0,
    "rejected": -1.0,
    "ignored":  -0.5,
    "cart_add": 3.0,
    "product_view": 1.0,
    "page_view": 0.5
}

def get_user_dynamic_intent(user_id: str, min_events: int = 2) -> dict:
    """Analyze recent user events strictly to override stale offline clusters."""
    log = _load_log()
    user_events = [e for e in log if str(e.get("user_id")) == str(user_id)]
    if len(user_events) < min_events:
        return None

    cat_scores = defaultdict(float)
    
    # Process last 80 actions for intent
    for ev in user_events[-80:]:
        action = ev.get("action", "")
        meta = ev.get("meta", {})
        cat = meta.get("category")
        
        # fallback if product.cat is inside meta.product instead
        if not cat and meta.get("product"):
            cat = meta["product"].get("cat")

        if cat:
            weight = REWARD_MAP.get(action, 0.5)
            cat_scores[cat] += weight

    if not cat_scores:
        return None

    # Get highest scoring category
    top_cat = max(cat_scores.items(), key=lambda x: x[1])[0]
    
    # Try mapping back to a cluster_id if possible
    intents = load_intents()
    best_cid = None
    for cid_str, info in (intents or {}).items():
        if info.get("name") == top_cat:
            best_cid = int(cid_str)
            break
            
    return {"category": top_cat, "cluster_id": best_cid, "score": cat_scores[top_cat]}


def log_feedback(
    user_id:   str,
    nudge_id:  str,
    action:    str,
    cluster_id: int = None,
    predicted_cluster: int = None,
    meta: dict = None
) -> dict:
    if action not in REWARD_MAP:
        # Accept custom actions like page_view, cart_add dynamically too
        if action not in ["page_view", "product_view", "cart_add"]:
            raise ValueError(f"Invalid action '{action}'.")

    reward = REWARD_MAP.get(action, 0.5)

    if cluster_id is None:
        cmap = load_cluster_map()
        cluster_id = cmap.get(str(user_id))

    if predicted_cluster is None:
        preds = get_predictions_by_user()
        pred  = preds.get(str(user_id), {})
        predicted_cluster = pred.get("predicted_cluster")

    event = {
        "user_id":           str(user_id),
        "nudge_id":          nudge_id,
        "action":            action,
        "reward":            reward,
        "cluster_id":        cluster_id,
        "predicted_cluster": predicted_cluster,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "meta":              meta or {}
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    existing = _load_log()
    existing.append(event)

    with open(LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    # FIX: Auto-compute weights after every N events (not just on explicit trigger)
    if len(existing) % 5 == 0:  # every 5 events update weights
        compute_reward_weights()

    return event


def _load_log() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ reward_log.json corrupted. Resetting log.")
        return []


def compute_reward_weights() -> dict:
    log = _load_log()
    if not log:
        return {}

    user_rewards = defaultdict(list)
    for event in log:
        user_rewards[event["user_id"]].append(event["reward"])

    raw_weights = {
        uid: sum(rewards) / len(rewards)
        for uid, rewards in user_rewards.items()
    }

    if raw_weights:
        mean_w = sum(raw_weights.values()) / len(raw_weights)
        mean_w = max(mean_w, 1e-8)
        weights = {uid: w / mean_w for uid, w in raw_weights.items()}
    else:
        weights = {}

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"✅ Reward weights computed for {len(weights)} users → {WEIGHTS_PATH}")
    return weights


def get_cluster_reward_stats() -> dict:
    log = _load_log()
    
    # FIX: Return summary even when log is empty
    if not log:
        return {
            "total_events": 0,
            "message": "No feedback logged yet. POST to /feedback to start.",
            "clusters": {}
        }

    cluster_stats = defaultdict(lambda: {"accepted": 0, "rejected": 0, "ignored": 0, "total": 0})

    for event in log:
        cid = event.get("cluster_id")
        if cid is not None:
            cluster_stats[cid][event["action"]] += 1
            cluster_stats[cid]["total"] += 1

    result = {}
    for cid, s in cluster_stats.items():
        total = s["total"] or 1
        result[cid] = {
            **s,
            "acceptance_rate": round(s["accepted"] / total, 3),
        }

    return {
        "total_events": len(log),
        "clusters": result,
        # FIX: Overall stats added
        "overall_acceptance_rate": round(
            sum(1 for e in log if e["action"] == "accepted") / len(log), 3
        ),
    }


# FIX: Threshold 100 → 10 (realistic for development/early prod)
def should_retrain(min_feedback_events: int = 10) -> bool:
    log = _load_log()
    return len(log) >= min_feedback_events


def trigger_retrain():
    if not should_retrain():
        log_count = len(_load_log())
        print(f"⏳ Not enough feedback yet ({log_count}/10 events). Keep collecting.")
        return False

    weights = compute_reward_weights()
    flag_path = os.path.join(BASE, "module4_Flowboost", "outputs", "retrain_flag.json")

    os.makedirs(os.path.dirname(flag_path), exist_ok=True)
    with open(flag_path, "w") as f:
        json.dump({
            "trigger":      True,
            "n_events":     len(_load_log()),
            "n_users":      len(weights),
            "weights_path": WEIGHTS_PATH,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"🔁 Retrain triggered → {flag_path}")
    return True