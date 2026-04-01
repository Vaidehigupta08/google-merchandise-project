"""
feedback_loop.py
================
Module 5 — Reward-Guided Feedback Loop (RG-CFM)

When a user accepts or rejects a nudge on the live site,
that signal is captured as a reward score and logged.

These reward scores are used to:
  1. Log and persist all feedback events
  2. Compute per-cluster reward statistics
  3. Trigger Module 4 model retraining with reward weights
     (Reward-Guided CFM = RG-CFM as described in temp.md)

Reward schema:
{
  "user_id":     "42",
  "nudge_id":    "nudge_42_1711234567",
  "action":      "accepted" | "rejected" | "ignored",
  "reward":      1.0 | 0.0 | 0.3,
  "cluster_id":  2,
  "predicted_cluster": 1,
  "timestamp":   "2026-03-30T12:00:00"
}
"""

import json
import os
import time
from datetime import datetime, timezone
from collections import defaultdict

from module5_agent.m5_data_loader import load_cluster_map, get_predictions_by_user


BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH     = os.path.join(BASE, "module5_agent", "outputs", "reward_log.json")
WEIGHTS_PATH = os.path.join(BASE, "module5_agent", "outputs", "reward_weights.json")


# ── Reward mapping ─────────────────────────────────────────────────────

REWARD_MAP = {
    "accepted": 1.0,    # user clicked nudge → strong positive signal
    "rejected": 0.0,    # user dismissed nudge → negative signal
    "ignored":  0.3,    # nudge shown but no interaction → weak signal
}


# ── Core feedback functions ────────────────────────────────────────────

def log_feedback(
    user_id:   str,
    nudge_id:  str,
    action:    str,           # "accepted" | "rejected" | "ignored"
    cluster_id: int = None,
    predicted_cluster: int = None,
) -> dict:
    """
    Record a single feedback event to reward_log.json.
    Called by the FastAPI endpoint when frontend reports user action.
    """
    if action not in REWARD_MAP:
        raise ValueError(f"Invalid action '{action}'. Must be: accepted, rejected, ignored")

    reward = REWARD_MAP[action]

    # Auto-resolve cluster_id if not provided
    if cluster_id is None:
        cmap = load_cluster_map()
        cluster_id = cmap.get(str(user_id))

    # Auto-resolve predicted_cluster if not provided
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
    }

    # Append to log file (thread-safe append)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    existing = _load_log()
    existing.append(event)

    with open(LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    return event


def _load_log() -> list:
    """Load existing reward log or return empty list."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return json.load(f)


# ── Reward weight computation (RG-CFM) ────────────────────────────────

def compute_reward_weights() -> dict:
    """
    Compute per-user importance weights from feedback history.
    These weights are passed to Module 4 retraining (RG-CFM).

    Formula:
        w(user) = avg(reward) across all their feedback events
        Normalized so mean weight = 1.0 across all users.

    Returns: {user_id: weight}
    """
    log = _load_log()
    if not log:
        return {}

    # Aggregate rewards per user
    user_rewards = defaultdict(list)
    for event in log:
        user_rewards[event["user_id"]].append(event["reward"])

    # Average reward per user
    raw_weights = {
        uid: sum(rewards) / len(rewards)
        for uid, rewards in user_rewards.items()
    }

    # Normalize: mean weight = 1.0
    if raw_weights:
        mean_w = sum(raw_weights.values()) / len(raw_weights)
        mean_w = max(mean_w, 1e-8)
        weights = {uid: w / mean_w for uid, w in raw_weights.items()}
    else:
        weights = {}

    # Save weights file (Module 4 train.py reads this)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"✅ Reward weights computed for {len(weights)} users → {WEIGHTS_PATH}")
    return weights


def get_cluster_reward_stats() -> dict:
    """
    Return per-cluster acceptance stats for monitoring.
    Useful for FastAPI /stats endpoint.
    """
    log = _load_log()
    if not log:
        return {}

    cluster_stats = defaultdict(lambda: {"accepted": 0, "rejected": 0, "ignored": 0, "total": 0})

    for event in log:
        cid = event.get("cluster_id")
        if cid is not None:
            cluster_stats[cid][event["action"]] += 1
            cluster_stats[cid]["total"] += 1

    # Add acceptance rate
    result = {}
    for cid, s in cluster_stats.items():
        total = s["total"] or 1
        result[cid] = {
            **s,
            "acceptance_rate": round(s["accepted"] / total, 3),
        }

    return result


# ── Retrain trigger ────────────────────────────────────────────────────

def should_retrain(min_feedback_events: int = 100) -> bool:
    """
    Returns True if enough new feedback has accumulated
    to warrant retraining Module 4.
    """
    log = _load_log()
    return len(log) >= min_feedback_events


def trigger_retrain():
    """
    Computes reward weights and writes a retrain_flag.json
    that Module 4 train.py checks before training.
    """
    if not should_retrain():
        print("⏳ Not enough feedback yet to retrain.")
        return False

    weights = compute_reward_weights()
    flag_path = os.path.join(BASE, "module4_Flowboost", "outputs", "retrain_flag.json")

    os.makedirs(os.path.dirname(flag_path), exist_ok=True)
    with open(flag_path, "w") as f:
        json.dump({
            "trigger":        True,
            "n_events":       len(_load_log()),
            "n_users":        len(weights),
            "weights_path":   WEIGHTS_PATH,
            "triggered_at":   datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"🔁 Retrain triggered → {flag_path}")
    return True
