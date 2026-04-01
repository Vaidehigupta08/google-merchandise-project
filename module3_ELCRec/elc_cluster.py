import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import defaultdict, Counter
from sklearn.cluster import KMeans


# =========================
# 🔹 ELCRec-style clustering model (FIXED ✅)
# =========================
class ELCCluster(nn.Module):
    def __init__(self, num_clusters, embedding_dim, init_centers):
        super().__init__()
        # Initialize with KMeans centers (NOT random samples)
        self.cluster_centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))

    def forward(self, x):
        # NOTE: x is already pre-normalized before passing in
        centers = nn.functional.normalize(self.cluster_centers, dim=1)
        sim = torch.matmul(x, centers.T)  # cosine similarity
        return sim


# =========================
# 🔹 Load embeddings
# =========================
def load_embeddings(path):
    with open(path, "r") as f:
        data = json.load(f)

    ids = list(data.keys())
    embeddings = np.array(list(data.values()), dtype=np.float32)

    # ✅ PRE-NORMALIZE embeddings (critical fix!)
    # Raw beha2vec embeddings have norms ~6-11, not unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"✅ Embeddings loaded: {embeddings.shape}")
    print(f"   Norm check after normalization: min={np.linalg.norm(embeddings, axis=1).min():.4f}, max={np.linalg.norm(embeddings, axis=1).max():.4f}")

    return ids, torch.tensor(embeddings)


# =========================
# 🔹 KMeans warm-start initialization (CRITICAL FIX ✅)
# =========================
def kmeans_init(embeddings_np, num_clusters, n_init=20):
    print(f"🔄 Running KMeans warm-start (n_init={n_init}) for better initial centers...")
    km = KMeans(n_clusters=num_clusters, n_init=n_init, max_iter=300, random_state=42)
    km.fit(embeddings_np)

    centers = km.cluster_centers_
    # Normalize the initial centers too
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

    dist = Counter(km.labels_)
    print(f"   KMeans initial distribution: {dict(sorted(dist.items()))}")
    return centers


# =========================
# 🔹 Train clustering model (BALANCED LOSS - FIXED ✅)
# =========================
def train_model(embeddings, num_clusters=5, epochs=200, lr=5e-4):
    device = "cpu"
    embeddings_np = embeddings.numpy()

    # ✅ KMeans warm-start (fixes cluster collapse)
    init_centers = kmeans_init(embeddings_np, num_clusters)

    model = ELCCluster(num_clusters, embeddings.shape[1], init_centers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    embeddings = embeddings.to(device)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        sim = model(embeddings)

        # Temperature-scaled softmax for sharper assignments
        temperature = 0.1
        probs = torch.softmax(sim / temperature, dim=1)

        # ✅ Confidence loss: maximize confidence of cluster assignment
        loss_conf = -torch.mean(torch.max(probs, dim=1)[0])

        # ✅ Balance loss: penalize skewed distribution (weight increased to 2.0)
        cluster_mean = torch.mean(probs, dim=0)
        loss_balance = torch.sum(cluster_mean * torch.log(cluster_mean + 1e-8))

        # ✅ Separation loss: push cluster centers apart
        centers_norm = nn.functional.normalize(model.cluster_centers, dim=1)
        sim_matrix = torch.matmul(centers_norm, centers_norm.T)
        mask = ~torch.eye(num_clusters, dtype=torch.bool)
        loss_sep = torch.mean(sim_matrix[mask])

        # Final loss with increased balance weight
        loss = loss_conf + 2.0 * loss_balance + 0.5 * loss_sep

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            with torch.no_grad():
                hard_assign = torch.argmax(sim, dim=1)
                dist = Counter(hard_assign.numpy().tolist())
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} "
                      f"(conf={loss_conf.item():.4f}, bal={loss_balance.item():.4f}, sep={loss_sep.item():.4f}) "
                      f"| Dist: {dict(sorted(dist.items()))}")

    # Load best model state
    model.load_state_dict(best_state)
    print(f"\n✅ Best loss: {best_loss:.4f}")
    return model


# =========================
# 🔹 Assign clusters
# =========================
def get_clusters(model, embeddings):
    with torch.no_grad():
        sim = model(embeddings)
        cluster_ids = torch.argmax(sim, dim=1)

    return cluster_ids.cpu().numpy()


# =========================
# 🔹 Extract latent intents
# =========================
def extract_intents(embeddings, cluster_ids, num_clusters):
    cluster_data = defaultdict(list)

    for idx, cid in enumerate(cluster_ids):
        cluster_data[cid].append(embeddings[idx].numpy())

    intents = {}

    for cid in range(num_clusters):
        cluster_vectors = np.array(cluster_data[cid])

        if len(cluster_vectors) == 0:
            intents[cid] = {
                "intent_vector": [],
                "size": 0
            }
            continue

        centroid = np.mean(cluster_vectors, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        intents[cid] = {
            "intent_vector": centroid.tolist(),
            "size": len(cluster_vectors)
        }

    return intents


# =========================
# 🔹 Save JSON
# =========================
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# =========================
# 🔹 MAIN PIPELINE
# =========================
if __name__ == "__main__":

    # ✅ Paths — adjust if running from different directory
    EMB_PATH = "../module2_beha2vec/embeddings.json"
    CLUSTER_OUT = "../module3_ELCRec/outputs/cluster_map.json"
    INTENT_OUT = "../module3_ELCRec/outputs/intents.json"

    NUM_CLUSTERS = 5

    print("🚀 Loading embeddings...")
    ids, embeddings = load_embeddings(EMB_PATH)

    print("\n🧠 Training ELC clustering...")
    model = train_model(embeddings, num_clusters=NUM_CLUSTERS, epochs=200, lr=5e-4)

    print("\n📊 Assigning clusters...")
    cluster_ids = get_clusters(model, embeddings)

    # Print final distribution
    dist = Counter(cluster_ids.tolist())
    print(f"\n✅ Final Cluster Distribution: {dict(sorted(dist.items()))}")

    # Check balance score (ideally each cluster ~1570/5 = 314 users)
    expected = len(ids) / NUM_CLUSTERS
    balance_ratio = min(dist.values()) / max(dist.values())
    print(f"   Expected per cluster: ~{expected:.0f}")
    print(f"   Balance ratio (min/max): {balance_ratio:.3f} (closer to 1.0 = better)")

    # Save cluster map
    cluster_map = {str(i): int(c) for i, c in zip(ids, cluster_ids)}
    save_json(cluster_map, CLUSTER_OUT)
    print(f"\n💾 Cluster map saved → {CLUSTER_OUT}")

    print("\n🔥 Extracting latent intents...")
    intents = extract_intents(embeddings, cluster_ids, num_clusters=NUM_CLUSTERS)
    save_json(intents, INTENT_OUT)
    print(f"💾 Intents saved → {INTENT_OUT}")

    print("\n✅ DONE! (2 FILES GENERATED: cluster_map.json + intents.json)")