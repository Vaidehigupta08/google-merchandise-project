"""
predict.py
==========
Module 4 — CFM Inference

Given a user's current state (embedding + cluster), predicts their
next embedding using the trained CFM model via ODE integration.

Integration method: Euler (fast) or RK4 (accurate)
Steps: 50 (default, can increase for accuracy)

Output:
    - module4_CFM/outputs/predictions.json
      [
        {
          "user_id": "42",
          "cluster_id": 2,
          "current_embedding": [...],
          "predicted_next_embedding": [...],
          "predicted_cluster": 1,
          "top_predicted_urls": ["Bags", "Clothing", "Electronics"]
        },
        ...
      ]
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict

from cfm_model import CFMVelocityNet


# ==========================
# Paths
# ==========================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH  = os.path.join(BASE, "module4_Flowboost", "outputs", "cfm_model.pt")
TRAJ_PATH   = os.path.join(BASE, "module4_Flowboost", "outputs", "trajectories.json")
INTENT_PATH = os.path.join(BASE, "module3_ELCRec", "outputs", "intents.json")
DATA_PATH   = os.path.join(BASE, "module2_beha2vec", "input", "data.csv")
OUT_PATH    = os.path.join(BASE, "module4_Flowboost", "outputs", "predictions.json")


# ==========================
# ODE Integration
# ==========================
@torch.no_grad()
def euler_integrate(model, x0, condition, steps=50):
    """
    Euler integration of the learned velocity field.
    Integrates from t=0 (noise) to t=1 (predicted embedding).

    x_{t+dt} = x_t + dt * v_theta(x_t, t, cond)
    """
    x = x0.clone()
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.float32)
        velocity = model(x, t, condition)
        x = x + dt * velocity

    return x


@torch.no_grad()
def rk4_integrate(model, x0, condition, steps=50):
    """
    RK4 integration — more accurate than Euler, same speed at low steps.
    """
    x = x0.clone()
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i * dt
        t_mid  = t_val + 0.5 * dt
        t_end  = t_val + dt

        t0 = torch.full((x.shape[0],), t_val,  device=x.device, dtype=torch.float32)
        tm = torch.full((x.shape[0],), t_mid,  device=x.device, dtype=torch.float32)
        t1 = torch.full((x.shape[0],), t_end,  device=x.device, dtype=torch.float32)

        k1 = model(x,                    t0, condition)
        k2 = model(x + 0.5 * dt * k1,   tm, condition)
        k3 = model(x + 0.5 * dt * k2,   tm, condition)
        k4 = model(x + dt * k3,          t1, condition)

        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x


# ==========================
# Nearest cluster lookup
# ==========================
def find_nearest_cluster(pred_emb: np.ndarray, intent_vectors: dict) -> int:
    """Find cluster whose intent vector is closest to predicted embedding."""
    best_cluster = 0
    best_sim = -1.0

    pred_norm = pred_emb / (np.linalg.norm(pred_emb) + 1e-8)

    for cid, vec in intent_vectors.items():
        v = np.array(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        sim = float(np.dot(pred_norm, v))
        if sim > best_sim:
            best_sim = sim
            best_cluster = int(cid)

    return best_cluster


# ==========================
# URL prediction from cluster
# ==========================
def predict_top_urls(cluster_id: int, cluster_url_stats: dict, top_k: int = 3) -> list:
    """Return top-k most visited URLs for a cluster."""
    url_counts = cluster_url_stats.get(cluster_id, {})
    if not url_counts:
        return []
    sorted_urls = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)
    return [url for url, _ in sorted_urls[:top_k]]


# ==========================
# Main prediction pipeline
# ==========================
def run_predictions(integration: str = "rk4", steps: int = 50, batch_size: int = 128):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = CFMVelocityNet(
        emb_dim    = cfg["emb_dim"],
        hidden_dim = cfg["hidden_dim"],
        time_dim   = cfg["time_dim"],
        num_layers = cfg["num_layers"],
        dropout    = cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"✅ Model loaded (trained {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f})")

    # Load trajectories
    with open(TRAJ_PATH) as f:
        trajectories = json.load(f)

    # Load intents
    with open(INTENT_PATH) as f:
        intents_raw = json.load(f)
    intent_vectors = {
        int(k): v["intent_vector"] for k, v in intents_raw.items()
    }

    # Build cluster -> URL stats from data.csv
    import pandas as pd
    from collections import Counter
    df = pd.read_csv(DATA_PATH)

    # Load cluster map
    cluster_map_path = os.path.join(BASE, "module3_ELCRec", "outputs", "cluster_map.json")
    with open(cluster_map_path) as f:
        cluster_map = json.load(f)

    cluster_url_stats = defaultdict(Counter)
    for _, row in df.iterrows():
        uid = str(row["user_id"])
        cid = cluster_map.get(uid)
        if cid is not None:
            cluster_url_stats[int(cid)][row["pageview_URL"]] += 1

    # Predict in batches
    print(f"\n🔮 Running {integration.upper()} integration ({steps} steps)...")
    predictions = []

    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i : i + batch_size]

        x1s   = torch.tensor([t["x1"] for t in batch],        dtype=torch.float32, device=device)
        conds = torch.tensor([t["condition"] for t in batch],  dtype=torch.float32, device=device)

        # Source noise (starting point for integration)
        x0 = torch.randn_like(x1s)

        # Integrate
        if integration == "euler":
            x_pred = euler_integrate(model, x0, conds, steps=steps)
        else:
            x_pred = rk4_integrate(model, x0, conds, steps=steps)

        x_pred_np = x_pred.cpu().numpy()

        for j, traj in enumerate(batch):
            pred_emb = x_pred_np[j]

            # Normalize predicted embedding
            pred_emb_norm = pred_emb / (np.linalg.norm(pred_emb) + 1e-8)

            # Find nearest cluster
            pred_cluster = find_nearest_cluster(pred_emb_norm, intent_vectors)

            # Top predicted URLs
            top_urls = predict_top_urls(pred_cluster, cluster_url_stats)

            predictions.append({
                "user_id":                   traj["user_id"],
                "current_cluster":           traj["cluster_id"],
                "current_url_sequence":      traj["url_sequence"],
                "predicted_next_embedding":  pred_emb_norm.tolist(),
                "predicted_cluster":         pred_cluster,
                "top_predicted_urls":        top_urls,
            })

        if (i // batch_size + 1) % 5 == 0:
            print(f"   Processed {min(i+batch_size, len(trajectories))}/{len(trajectories)}")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\n✅ Predictions saved → {OUT_PATH}")
    print(f"   Total users predicted: {len(predictions)}")

    # Summary
    from collections import Counter
    trans = Counter(
        (p["current_cluster"], p["predicted_cluster"])
        for p in predictions
    )
    print("\n📊 Cluster transition summary (current → predicted):")
    for (src, dst), count in sorted(trans.items()):
        print(f"   Cluster {src} → Cluster {dst}: {count} users")

    return predictions


if __name__ == "__main__":
    run_predictions(integration="rk4", steps=50)
