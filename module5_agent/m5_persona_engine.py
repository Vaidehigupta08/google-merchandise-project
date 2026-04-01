"""
persona_engine.py
=================
Module 5 — Buyer Persona Generator

For each cluster, sends behavioral data to LLM and gets back
a structured buyer persona. Personas are cached to disk so
LLM is only called once per cluster (or on demand refresh).

Output schema per cluster:
{
  "cluster_id": 2,
  "persona_name": "The Fashion Explorer",
  "age_range": "22-35",
  "shopping_style": "...",
  "top_interests": ["Bags", "Clothing"],
  "purchase_intent": "high | medium | low",
  "nudge_strategy": "...",
  "pain_points": "...",
  "recommended_offers": ["...", "..."]
}
"""

import json
import os
from collections import Counter

from module5_agent.m5_data_loader import (
    load_intents,
    load_cluster_map,
    get_cluster_url_stats,
    get_cluster_sizes,
    load_clickstream,
)
from module5_agent.m5_llm_client import llm_call


BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE, "module5_agent", "outputs", "personas.json")


# ── Prompt builder ─────────────────────────────────────────────────────

def _build_prompt(cluster_id: int, stats: dict) -> tuple[str, str]:
    url_stats   = get_cluster_url_stats()
    sizes       = get_cluster_sizes()
    url_counts  = url_stats.get(cluster_id, {})
    total_clicks = sum(url_counts.values()) or 1
    size         = sizes.get(cluster_id, 0)

    # Top URLs with percentages
    top_urls = sorted(url_counts.items(), key=lambda x: x[1], reverse=True)
    url_lines = "\n".join(
        f"  - {url}: {cnt} clicks ({cnt/total_clicks*100:.1f}%)"
        for url, cnt in top_urls
    )

    # Average session stats
    df         = load_clickstream()
    cmap       = load_cluster_map()
    user_ids   = [int(k) for k, v in cmap.items() if v == cluster_id]
    user_df    = df[df["user_id"].isin(user_ids)]
    avg_len    = user_df.groupby("user_id").size().mean() if len(user_df) > 0 else 0
    seq_counts = user_df.groupby("user_id").size()
    long_sess  = int((seq_counts >= 8).sum())

    system_prompt = (
        "You are an expert e-commerce analyst. "
        "Given behavioral data about a user cluster, generate a structured buyer persona. "
        "Always respond with valid JSON only — no markdown, no explanation."
    )

    user_prompt = f"""
Analyze this e-commerce user cluster and generate a buyer persona.

Cluster ID: {cluster_id}
Total users: {size}
Average session length: {avg_len:.1f} page views
Users with long sessions (8+ pages): {long_sess}

Page category visit distribution:
{url_lines}

Return a JSON object with these exact fields:
{{
  "cluster_id": {cluster_id},
  "persona_name": "a creative 3-word name for this shopper type",
  "age_range": "estimated age range like 22-35",
  "shopping_style": "2 sentences describing how this person shops",
  "top_interests": ["list", "of", "3", "categories"],
  "purchase_intent": "high or medium or low",
  "nudge_strategy": "1 sentence — what kind of message would convert this user",
  "pain_points": "1 sentence — what friction might stop them from buying",
  "recommended_offers": ["offer idea 1", "offer idea 2"]
}}
"""
    return system_prompt, user_prompt


# ── Persona generation ─────────────────────────────────────────────────

def generate_persona(cluster_id: int) -> dict:
    """Generate persona for a single cluster via LLM."""
    system_prompt, user_prompt = _build_prompt(cluster_id, {})
    raw = llm_call(system_prompt, user_prompt, max_tokens=600)

    try:
        persona = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON substring if LLM added extra text
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            persona = json.loads(match.group())
        else:
            persona = {
                "cluster_id":        cluster_id,
                "persona_name":      f"Cluster {cluster_id} Shopper",
                "raw_response":      raw,
                "error":             "LLM did not return valid JSON"
            }

    persona["cluster_id"] = cluster_id  # ensure correct
    return persona


def generate_all_personas(force_refresh: bool = False) -> list:
    """
    Generate personas for all clusters.
    Uses cached personas.json unless force_refresh=True.
    """
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load existing cache
    existing = {}
    if os.path.exists(OUT_PATH) and not force_refresh:
        with open(OUT_PATH) as f:
            cached = json.load(f)
        existing = {p["cluster_id"]: p for p in cached}
        print(f"📋 Loaded {len(existing)} cached personas")

    intents    = load_intents()
    all_ids    = [int(k) for k in intents.keys()]
    personas   = []
    generated  = 0

    for cid in sorted(all_ids):
        if cid in existing and not force_refresh:
            personas.append(existing[cid])
            print(f"   ↩️  Cluster {cid}: using cache")
        else:
            print(f"   🤖 Cluster {cid}: calling LLM...")
            persona = generate_persona(cid)
            personas.append(persona)
            generated += 1

    with open(OUT_PATH, "w") as f:
        json.dump(personas, f, indent=2)

    print(f"✅ Personas saved → {OUT_PATH}  ({generated} newly generated)")
    return personas


def load_personas() -> list:
    """Load cached personas from disk."""
    if not os.path.exists(OUT_PATH):
        return []
    with open(OUT_PATH) as f:
        return json.load(f)


def get_persona_for_cluster(cluster_id: int) -> dict | None:
    personas = load_personas()
    for p in personas:
        if p.get("cluster_id") == cluster_id:
            return p
    return None
