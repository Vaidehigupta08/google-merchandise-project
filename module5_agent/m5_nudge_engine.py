"""
nudge_engine.py
===============
Module 5 — Proactive Nudge Generator

Given a user's current state + predicted next state (from Module 4),
generates a personalized UI nudge (popup text + offer).

Nudge schema:
{
  "user_id": "42",
  "current_cluster": 2,
  "predicted_cluster": 1,
  "nudge_text": "Looks like you're exploring Bags! Check out our new arrivals.",
  "offer": "Get 10% off Bags today — use code BAGS10",
  "cta": "Shop Bags Now",
  "priority": "high | medium | low",
  "predicted_urls": ["Bags", "Clothing"]
}

Two nudge modes:
  1. LLM mode   — personalized text via LLM API
  2. Rule mode  — fast template-based nudges (no API needed)
     Used when LLM_PROVIDER=placeholder OR for high-volume users.
"""

import json
import os
from collections import Counter

from module5_agent.m5_data_loader import (
    load_predictions,
    get_predictions_by_user,
    get_cluster_url_stats,
    load_cluster_map,
)
from module5_agent.m5_persona_engine import get_persona_for_cluster
from module5_agent.m5_llm_client import llm_call, LLM_PROVIDER


BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE, "module5_agent", "outputs", "nudge_cache.json")


# ── Rule-based nudge templates (no LLM needed) ────────────────────────
# Multiple variants per category — picked deterministically per user_id
# so same user always gets same nudge, but different users get different ones.

_TEMPLATES = {
    "Bags": {
        "texts": [
            "Spotted a bag lover! 🎒 New arrivals just dropped.",
            "Your next favourite bag is waiting — check it out! 🎒",
            "Bags that turn heads — find yours today! 🛍️",
            "Complete your look with the perfect bag! 🎒",
        ],
        "offers": [
            "Free shipping on all Bags today",
            "Buy any Bag, get a free pouch",
            "Up to 15% off Bags — limited time",
            "Extra ₹200 off on Bags over ₹999",
        ],
        "ctas": ["Shop Bags", "View Bags", "Explore Bags", "Find My Bag"],
    },
    "Clothing": {
        "texts": [
            "New styles just dropped — don't miss out! 👗",
            "Your wardrobe called, it wants an upgrade! 👕",
            "Fresh arrivals in Clothing — just for you! ✨",
            "Trending styles you'll love are here! 👗",
        ],
        "offers": [
            "15% off Clothing — today only",
            "Buy 2, get 1 free on all Clothing",
            "Flat ₹300 off on orders above ₹1499",
            "Free delivery on Clothing this weekend",
        ],
        "ctas": ["Browse Clothing", "Shop Styles", "View Collection", "Explore Now"],
    },
    "Accessories": {
        "texts": [
            "Complete your look with the right accessories! ✨",
            "Small details, big impact — shop Accessories! 💍",
            "The finishing touch your outfit needs is here! ✨",
            "Accessorise your way — new picks available! 💎",
        ],
        "offers": [
            "Buy 2 Accessories, get 1 free",
            "20% off on all Accessories today",
            "Free gift wrap on Accessories over ₹599",
            "Combo deal: any 3 Accessories at ₹999",
        ],
        "ctas": ["Shop Accessories", "Complete My Look", "View Picks", "Explore Accessories"],
    },
    "Electronics": {
        "texts": [
            "Tech explorer detected! Deals just for you 💻",
            "Level up your setup — new Electronics in stock! 🖥️",
            "The gadget you've been eyeing is on sale! ⚡",
            "Smart picks for smart shoppers — check Electronics! 💡",
        ],
        "offers": [
            "Up to 20% off Electronics",
            "No-cost EMI on Electronics above ₹2999",
            "Free accessories worth ₹499 with Electronics",
            "Flat ₹500 off on Electronics this week",
        ],
        "ctas": ["Explore Tech", "Shop Electronics", "View Deals", "See Offers"],
    },
    "Drinkware": {
        "texts": [
            "Stay hydrated in style — new Drinkware here! ☕",
            "Your perfect cup is just a click away! 🍵",
            "Sip smarter — explore our Drinkware range! ☕",
            "New tumblers & mugs just arrived! 🧃",
        ],
        "offers": [
            "Free tumbler with ₹999+ order",
            "Buy 2 Drinkware items, get 15% off",
            "Free shipping on all Drinkware",
            "Flat ₹150 off on Drinkware combos",
        ],
        "ctas": ["Shop Drinkware", "View Tumblers", "Explore Mugs", "Shop Now"],
    },
    "Office": {
        "texts": [
            "Upgrade your workspace — new Office picks! 🖊️",
            "Work smarter with our Office essentials! 💼",
            "Your desk deserves an upgrade — check this out! 🖥️",
            "Productivity boosters are on sale today! 📋",
        ],
        "offers": [
            "Office essentials on sale — up to 25% off",
            "Buy Office items worth ₹799+, get free delivery",
            "Flat 10% off on all Office accessories",
            "Bundle deal: 3 Office items at special price",
        ],
        "ctas": ["Shop Office", "Upgrade Workspace", "View Essentials", "Explore Office"],
    },
    "Kids": {
        "texts": [
            "Little ones deserve the best — shop Kids now! 🧸",
            "Fun finds for your little ones are here! 🎈",
            "New arrivals in Kids — they'll love it! 🧸",
            "Make their day special — explore Kids section! 🎁",
        ],
        "offers": [
            "10% off all Kids items today",
            "Buy 2 Kids items, get 1 free",
            "Free gift wrapping on Kids orders",
            "Flat ₹200 off on Kids above ₹999",
        ],
        "ctas": ["Shop Kids", "View Kids", "Explore Kids", "Find Gifts"],
    },
    "default": {
        "texts": [
            "Something special picked just for you! 🎁",
            "Exclusive deals are waiting — have a look! ✨",
            "Don't miss today's top picks for you! 🛍️",
            "We found something you'll love! 🎉",
        ],
        "offers": [
            "Exclusive offer inside — grab it now",
            "Today's deal: extra 10% off sitewide",
            "Limited-time offer just for you",
            "Special discount on your next order",
        ],
        "ctas": ["View Offer", "See Deals", "Explore Now", "Shop Now"],
    },
}

# Transition-aware texts: when user is moving to a NEW cluster (high priority)
_TRANSITION_PREFIX = {
    "Bags":        "Looks like you're shifting to Bags — ",
    "Clothing":    "Exploring Clothing next? — ",
    "Accessories": "Moving toward Accessories — ",
    "Electronics": "Heading into Electronics? — ",
    "Drinkware":   "Eyeing Drinkware now — ",
    "Office":      "Checking out Office items — ",
    "Kids":        "Browsing Kids section now — ",
}


def _rule_nudge(user_id: str, prediction: dict) -> dict:
    """
    Template-based nudge with per-user variation.
    Uses user_id as seed so same user always gets same nudge,
    but different users get different text/offer/cta variants.
    """
    import hashlib

    top_urls    = prediction.get("top_predicted_urls", [])
    url_seq     = prediction.get("current_url_sequence", [])
    curr        = prediction.get("current_cluster", -1)
    pred_cluster = prediction.get("predicted_cluster", -1)
    priority    = "high" if curr != pred_cluster else "medium"

    # Pick primary URL — prefer predicted, fallback to current browsing
    url = top_urls[0] if top_urls else "default"

    # Deterministic index from user_id hash — ensures same user = same nudge
    uid_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    bucket   = _TEMPLATES.get(url, _TEMPLATES["default"])
    idx      = uid_hash % len(bucket["texts"])

    text  = bucket["texts"][idx]
    offer = bucket["offers"][idx % len(bucket["offers"])]
    cta   = bucket["ctas"][idx % len(bucket["ctas"])]

    # High-priority (cluster transition): prepend context-aware prefix
    if priority == "high" and url in _TRANSITION_PREFIX:
        text = _TRANSITION_PREFIX[url] + text[0].lower() + text[1:]

    # If user browsed multiple categories, mention where they came from
    if url_seq and len(url_seq) >= 2:
        from_url = url_seq[-2] if url_seq[-1] == url else url_seq[-1]
        if from_url and from_url != url and from_url != "default":
            # Add context only for medium priority (not transitioning users — already have prefix)
            if priority == "medium":
                text = text.rstrip(".!") + f" — also loved in {from_url}!"

    # Unique nudge_id per user (stable, not timestamp-based)
    nudge_id = f"nudge_{user_id}_{uid_hash % 99999:05d}"

    return {
        "user_id":           str(user_id),
        "nudge_id":          nudge_id,
        "current_cluster":   curr,
        "predicted_cluster": pred_cluster,
        "nudge_text":        text,
        "offer":             offer,
        "cta":               cta,
        "priority":          priority,
        "predicted_urls":    top_urls,
        "mode":              "rule",
    }


# ── LLM-based nudge ───────────────────────────────────────────────────

def _llm_nudge(user_id: str, prediction: dict) -> dict:
    """Personalized nudge via LLM."""
    persona   = get_persona_for_cluster(prediction.get("current_cluster", 0))
    top_urls  = prediction.get("top_predicted_urls", [])
    url_seq   = prediction.get("current_url_sequence", [])
    pred_cid  = prediction.get("predicted_cluster", 0)
    pred_p    = get_persona_for_cluster(pred_cid)

    persona_name   = persona.get("persona_name", "Shopper") if persona else "Shopper"
    nudge_strategy = persona.get("nudge_strategy", "") if persona else ""
    pred_style     = pred_p.get("persona_name", "") if pred_p else ""

    system_prompt = (
        "You are a conversion optimization expert for an e-commerce store. "
        "Write a short, friendly nudge message to show as a popup. "
        "Respond only with JSON — no markdown."
    )
    user_prompt = f"""
User browsed: {' → '.join(url_seq[-5:])}
Their buyer type: {persona_name}
Predicted next interest: {', '.join(top_urls[:2])}
Moving toward: {pred_style}
Nudge strategy for this user: {nudge_strategy}

Return JSON:
{{
  "nudge_text": "one friendly sentence (max 15 words)",
  "offer": "one specific offer string",
  "cta": "2-3 word button text"
}}
"""
    raw = llm_call(system_prompt, user_prompt, max_tokens=200)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data  = json.loads(match.group()) if match else {}

    curr = prediction.get("current_cluster", -1)
    pred = prediction.get("predicted_cluster", -1)

    import hashlib
    uid_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    nudge_id = f"nudge_{user_id}_{uid_hash % 99999:05d}"

    return {
        "user_id":           str(user_id),
        "nudge_id":          nudge_id,
        "current_cluster":   curr,
        "predicted_cluster": pred,
        "nudge_text":        data.get("nudge_text", "Special offer just for you!"),
        "offer":             data.get("offer", "Check our latest deals"),
        "cta":               data.get("cta", "View Now"),
        "priority":          "high" if curr != pred else "medium",
        "predicted_urls":    top_urls,
        "mode":              "llm",
    }


# ── Public API ────────────────────────────────────────────────────────

def _generate_override_nudge(user_id: str, override: dict) -> dict:
    """Mock a prediction object out of override intent to get a dynamic nudge."""
    prediction = {
        "user_id": user_id,
        "predicted_category": override.get("category"),
        "predicted_cluster": override.get("cluster_id")
    }

    if LLM_PROVIDER == "placeholder":
        nudge = _rule_nudge(user_id, prediction)
    else:
        try:
            nudge = _llm_nudge(user_id, prediction)
        except Exception:
            nudge = _rule_nudge(user_id, prediction)
    
    nudge["mode"] = "dynamic_override"
    return nudge


def get_nudge_for_user(user_id: str, override: dict = None) -> dict:
    """
    Get nudge for a single user.
    1. If override is provided, generate a nudge dynamically based on override.
    2. Check nudge_cache.json first (pre-generated)
    3. If not cached, generate on-the-fly from predictions
    4. If no prediction found, return default nudge
    """
    if override:
        return _generate_override_nudge(user_id, override)

    cache = load_nudge_cache()
    nudge = cache.get(str(user_id))

    if nudge:
        return nudge

    # Cache miss — generate on-the-fly from predictions
    predictions = get_predictions_by_user()
    prediction  = predictions.get(str(user_id))

    if not prediction:
        return {
            "user_id":    user_id,
            "nudge_text": "Welcome! Explore our latest collection.",
            "offer":      "10% off your first order",
            "cta":        "Shop Now",
            "priority":   "low",
            "mode":       "default",
        }

    if LLM_PROVIDER == "placeholder":
        return _rule_nudge(user_id, prediction)
    else:
        try:
            return _llm_nudge(user_id, prediction)
        except Exception as e:
            print(f"⚠️  LLM nudge failed for user {user_id}: {e}. Using rule fallback.")
            return _rule_nudge(user_id, prediction)


def generate_all_nudges(use_llm: bool = False) -> list:
    """
    Pre-generate nudges for ALL users and cache to disk.
    Call this once after Module 4 predictions are ready.
    """
    predictions = load_predictions()
    nudges      = []

    print(f"🔔 Generating nudges for {len(predictions)} users...")

    for i, pred in enumerate(predictions):
        uid = pred["user_id"]
        if use_llm and LLM_PROVIDER != "placeholder":
            nudge = _llm_nudge(uid, pred)
        else:
            nudge = _rule_nudge(uid, pred)
        nudges.append(nudge)

        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(predictions)} done...")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(nudges, f, indent=2)

    # Summary
    prio = Counter(n["priority"] for n in nudges)
    trans = Counter(
        f"{n['current_cluster']}→{n['predicted_cluster']}"
        for n in nudges
    )
    print(f"✅ Nudges saved → {OUT_PATH}")
    print(f"   Priority breakdown: {dict(prio)}")
    print(f"   Top transitions: {dict(trans.most_common(5))}")
    return nudges


def load_nudge_cache() -> dict:
    """Returns {user_id: nudge_dict} from cached file."""
    if not os.path.exists(OUT_PATH):
        return {}
    with open(OUT_PATH) as f:
        nudges = json.load(f)
    return {n["user_id"]: n for n in nudges}
# """
# nudge_engine.py
# ===============
# Module 5 — Proactive Nudge Generator

# Given a user's current state + predicted next state (from Module 4),
# generates a personalized UI nudge (popup text + offer).

# Nudge schema:
# {
#   "user_id": "42",
#   "current_cluster": 2,
#   "predicted_cluster": 1,
#   "nudge_text": "Looks like you're exploring Bags! Check out our new arrivals.",
#   "offer": "Get 10% off Bags today — use code BAGS10",
#   "cta": "Shop Bags Now",
#   "priority": "high | medium | low",
#   "predicted_urls": ["Bags", "Clothing"]
# }

# Two nudge modes:
#   1. LLM mode   — personalized text via LLM API
#   2. Rule mode  — fast template-based nudges (no API needed)
#      Used when LLM_PROVIDER=placeholder OR for high-volume users.
# """

# import json
# import os
# from collections import Counter

# from module5_agent.m5_data_loader import (
#     load_predictions,
#     get_predictions_by_user,
#     get_cluster_url_stats,
#     load_cluster_map,
# )
# from module5_agent.m5_persona_engine import get_persona_for_cluster
# from module5_agent.m5_llm_client import llm_call, LLM_PROVIDER


# BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# OUT_PATH = os.path.join(BASE, "module5_agent", "outputs", "nudge_cache.json")


# # ── Rule-based nudge templates (no LLM needed) ────────────────────────

# _TEMPLATES = {
#     "Bags":        ("Spotted a bag lover! 🎒", "Free shipping on all Bags today",    "Shop Bags"),
#     "Clothing":    ("New styles just dropped! 👗", "15% off Clothing — today only",  "Browse Clothing"),
#     "Accessories": ("Complete your look! ✨",       "Buy 2 Accessories, get 1 free", "Shop Accessories"),
#     "Electronics": ("Tech explorer detected! 💻",   "Up to 20% off Electronics",     "Explore Tech"),
#     "Drinkware":   ("Stay hydrated in style! ☕",   "Free tumbler with ₹999+ order", "Shop Drinkware"),
#     "Office":      ("Upgrade your workspace! 🖊️",  "Office essentials on sale",     "Shop Office"),
#     "Kids":        ("Little ones deserve the best! 🧸", "10% off Kids items",        "Shop Kids"),
#     "default":     ("Something special for you! 🎁", "Exclusive offer inside",       "View Offer"),
# }

# def _rule_nudge(user_id: str, prediction: dict) -> dict:
#     """Fast template-based nudge — no LLM call."""
#     top_urls = prediction.get("top_predicted_urls", [])
#     url      = top_urls[0] if top_urls else "default"
#     text, offer, cta = _TEMPLATES.get(url, _TEMPLATES["default"])

#     # Priority based on cluster transition
#     curr = prediction.get("current_cluster", -1)
#     pred = prediction.get("predicted_cluster", -1)
#     priority = "high" if curr != pred else "medium"

#     return {
#         "user_id":          user_id,
#         "current_cluster":  curr,
#         "predicted_cluster": pred,
#         "nudge_text":       text,
#         "offer":            offer,
#         "cta":              cta,
#         "priority":         priority,
#         "predicted_urls":   top_urls,
#         "mode":             "rule",
#     }


# # ── LLM-based nudge ───────────────────────────────────────────────────

# def _llm_nudge(user_id: str, prediction: dict) -> dict:
#     """Personalized nudge via LLM."""
#     persona   = get_persona_for_cluster(prediction.get("current_cluster", 0))
#     top_urls  = prediction.get("top_predicted_urls", [])
#     url_seq   = prediction.get("current_url_sequence", [])
#     pred_cid  = prediction.get("predicted_cluster", 0)
#     pred_p    = get_persona_for_cluster(pred_cid)

#     persona_name   = persona.get("persona_name", "Shopper") if persona else "Shopper"
#     nudge_strategy = persona.get("nudge_strategy", "") if persona else ""
#     pred_style     = pred_p.get("persona_name", "") if pred_p else ""

#     system_prompt = (
#         "You are a conversion optimization expert for an e-commerce store. "
#         "Write a short, friendly nudge message to show as a popup. "
#         "Respond only with JSON — no markdown."
#     )
#     user_prompt = f"""
# User browsed: {' → '.join(url_seq[-5:])}
# Their buyer type: {persona_name}
# Predicted next interest: {', '.join(top_urls[:2])}
# Moving toward: {pred_style}
# Nudge strategy for this user: {nudge_strategy}

# Return JSON:
# {{
#   "nudge_text": "one friendly sentence (max 15 words)",
#   "offer": "one specific offer string",
#   "cta": "2-3 word button text"
# }}
# """
#     raw = llm_call(system_prompt, user_prompt, max_tokens=200)

#     try:
#         data = json.loads(raw)
#     except json.JSONDecodeError:
#         import re
#         match = re.search(r'\{.*\}', raw, re.DOTALL)
#         data  = json.loads(match.group()) if match else {}

#     curr = prediction.get("current_cluster", -1)
#     pred = prediction.get("predicted_cluster", -1)

#     return {
#         "user_id":           user_id,
#         "current_cluster":   curr,
#         "predicted_cluster": pred,
#         "nudge_text":        data.get("nudge_text", "Special offer just for you!"),
#         "offer":             data.get("offer", "Check our latest deals"),
#         "cta":               data.get("cta", "View Now"),
#         "priority":          "high" if curr != pred else "medium",
#         "predicted_urls":    top_urls,
#         "mode":              "llm",
#     }


# # ── Public API ────────────────────────────────────────────────────────

# def get_nudge_for_user(user_id: str) -> dict:
#     """
#     Get nudge for a single user.
#     1. Check nudge_cache.json first (pre-generated)
#     2. If not cached, generate on-the-fly from predictions
#     3. If no prediction found, return default nudge
#     """
#     cache = load_nudge_cache()
#     nudge = cache.get(str(user_id))

#     if nudge:
#         return nudge

#     # Cache miss — generate on-the-fly from predictions
#     predictions = get_predictions_by_user()
#     prediction  = predictions.get(str(user_id))

#     if not prediction:
#         return {
#             "user_id":    user_id,
#             "nudge_text": "Welcome! Explore our latest collection.",
#             "offer":      "10% off your first order",
#             "cta":        "Shop Now",
#             "priority":   "low",
#             "mode":       "default",
#         }

#     if LLM_PROVIDER == "placeholder":
#         return _rule_nudge(user_id, prediction)
#     else:
#         try:
#             return _llm_nudge(user_id, prediction)
#         except Exception as e:
#             print(f"⚠️  LLM nudge failed for user {user_id}: {e}. Using rule fallback.")
#             return _rule_nudge(user_id, prediction)


# def generate_all_nudges(use_llm: bool = False) -> list:
#     """
#     Pre-generate nudges for ALL users and cache to disk.
#     Call this once after Module 4 predictions are ready.
#     """
#     predictions = load_predictions()
#     nudges      = []

#     print(f"🔔 Generating nudges for {len(predictions)} users...")

#     for i, pred in enumerate(predictions):
#         uid = pred["user_id"]
#         if use_llm and LLM_PROVIDER != "placeholder":
#             nudge = _llm_nudge(uid, pred)
#         else:
#             nudge = _rule_nudge(uid, pred)
#         nudges.append(nudge)

#         if (i + 1) % 200 == 0:
#             print(f"   {i+1}/{len(predictions)} done...")

#     os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
#     with open(OUT_PATH, "w") as f:
#         json.dump(nudges, f, indent=2)

#     # Summary
#     prio = Counter(n["priority"] for n in nudges)
#     trans = Counter(
#         f"{n['current_cluster']}→{n['predicted_cluster']}"
#         for n in nudges
#     )
#     print(f"✅ Nudges saved → {OUT_PATH}")
#     print(f"   Priority breakdown: {dict(prio)}")
#     print(f"   Top transitions: {dict(trans.most_common(5))}")
#     return nudges


# def load_nudge_cache() -> dict:
#     """Returns {user_id: nudge_dict} from cached file."""
#     if not os.path.exists(OUT_PATH):
#         return {}
#     with open(OUT_PATH) as f:
#         nudges = json.load(f)
#     return {n["user_id"]: n for n in nudges}

# # """
# # nudge_engine.py
# # ===============
# # Module 5 — Proactive Nudge Generator

# # Given a user's current state + predicted next state (from Module 4),
# # generates a personalized UI nudge (popup text + offer).

# # Nudge schema:
# # {
# #   "user_id": "42",
# #   "current_cluster": 2,
# #   "predicted_cluster": 1,
# #   "nudge_text": "Looks like you're exploring Bags! Check out our new arrivals.",
# #   "offer": "Get 10% off Bags today — use code BAGS10",
# #   "cta": "Shop Bags Now",
# #   "priority": "high | medium | low",
# #   "predicted_urls": ["Bags", "Clothing"]
# # }

# # Two nudge modes:
# #   1. LLM mode   — personalized text via LLM API
# #   2. Rule mode  — fast template-based nudges (no API needed)
# #      Used when LLM_PROVIDER=placeholder OR for high-volume users.
# # """

# # import json
# # import os
# # from collections import Counter

# # from module5_agent.m5_data_loader import (
# #     load_predictions,
# #     get_predictions_by_user,
# #     get_cluster_url_stats,
# #     load_cluster_map,
# # )
# # from module5_agent.m5_persona_engine import get_persona_for_cluster
# # from module5_agent.m5_llm_client import llm_call, LLM_PROVIDER


# # BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # OUT_PATH = os.path.join(BASE, "module5_agent", "outputs", "nudge_cache.json")


# # # ── Rule-based nudge templates (no LLM needed) ────────────────────────

# # _TEMPLATES = {
# #     "Bags":        ("Spotted a bag lover! 🎒", "Free shipping on all Bags today",    "Shop Bags"),
# #     "Clothing":    ("New styles just dropped! 👗", "15% off Clothing — today only",  "Browse Clothing"),
# #     "Accessories": ("Complete your look! ✨",       "Buy 2 Accessories, get 1 free", "Shop Accessories"),
# #     "Electronics": ("Tech explorer detected! 💻",   "Up to 20% off Electronics",     "Explore Tech"),
# #     "Drinkware":   ("Stay hydrated in style! ☕",   "Free tumbler with ₹999+ order", "Shop Drinkware"),
# #     "Office":      ("Upgrade your workspace! 🖊️",  "Office essentials on sale",     "Shop Office"),
# #     "Kids":        ("Little ones deserve the best! 🧸", "10% off Kids items",        "Shop Kids"),
# #     "default":     ("Something special for you! 🎁", "Exclusive offer inside",       "View Offer"),
# # }

# # def _rule_nudge(user_id: str, prediction: dict) -> dict:
# #     """Fast template-based nudge — no LLM call."""
# #     top_urls = prediction.get("top_predicted_urls", [])
# #     url      = top_urls[0] if top_urls else "default"
# #     text, offer, cta = _TEMPLATES.get(url, _TEMPLATES["default"])

# #     # Priority based on cluster transition
# #     curr = prediction.get("current_cluster", -1)
# #     pred = prediction.get("predicted_cluster", -1)
# #     priority = "high" if curr != pred else "medium"

# #     return {
# #         "user_id":          user_id,
# #         "current_cluster":  curr,
# #         "predicted_cluster": pred,
# #         "nudge_text":       text,
# #         "offer":            offer,
# #         "cta":              cta,
# #         "priority":         priority,
# #         "predicted_urls":   top_urls,
# #         "mode":             "rule",
# #     }


# # # ── LLM-based nudge ───────────────────────────────────────────────────

# # def _llm_nudge(user_id: str, prediction: dict) -> dict:
# #     """Personalized nudge via LLM."""
# #     persona   = get_persona_for_cluster(prediction.get("current_cluster", 0))
# #     top_urls  = prediction.get("top_predicted_urls", [])
# #     url_seq   = prediction.get("current_url_sequence", [])
# #     pred_cid  = prediction.get("predicted_cluster", 0)
# #     pred_p    = get_persona_for_cluster(pred_cid)

# #     persona_name   = persona.get("persona_name", "Shopper") if persona else "Shopper"
# #     nudge_strategy = persona.get("nudge_strategy", "") if persona else ""
# #     pred_style     = pred_p.get("persona_name", "") if pred_p else ""

# #     system_prompt = (
# #         "You are a conversion optimization expert for an e-commerce store. "
# #         "Write a short, friendly nudge message to show as a popup. "
# #         "Respond only with JSON — no markdown."
# #     )
# #     user_prompt = f"""
# # User browsed: {' → '.join(url_seq[-5:])}
# # Their buyer type: {persona_name}
# # Predicted next interest: {', '.join(top_urls[:2])}
# # Moving toward: {pred_style}
# # Nudge strategy for this user: {nudge_strategy}

# # Return JSON:
# # {{
# #   "nudge_text": "one friendly sentence (max 15 words)",
# #   "offer": "one specific offer string",
# #   "cta": "2-3 word button text"
# # }}
# # """
# #     raw = llm_call(system_prompt, user_prompt, max_tokens=200)

# #     try:
# #         data = json.loads(raw)
# #     except json.JSONDecodeError:
# #         import re
# #         match = re.search(r'\{.*\}', raw, re.DOTALL)
# #         data  = json.loads(match.group()) if match else {}

# #     curr = prediction.get("current_cluster", -1)
# #     pred = prediction.get("predicted_cluster", -1)

# #     return {
# #         "user_id":           user_id,
# #         "current_cluster":   curr,
# #         "predicted_cluster": pred,
# #         "nudge_text":        data.get("nudge_text", "Special offer just for you!"),
# #         "offer":             data.get("offer", "Check our latest deals"),
# #         "cta":               data.get("cta", "View Now"),
# #         "priority":          "high" if curr != pred else "medium",
# #         "predicted_urls":    top_urls,
# #         "mode":              "llm",
# #     }


# # # ── Public API ────────────────────────────────────────────────────────

# # def get_nudge_for_user(user_id: str) -> dict:
# #     cache = load_nudge_cache()
# #     nudge = cache.get(str(user_id))

# #     if nudge:
# #         return nudge

# #     return {
# #         "user_id":    user_id,
# #         "nudge_text": "Welcome! Explore our latest collection.",
# #         "offer":      "10% off your first order",
# #         "cta":        "Shop Now",
# #         "priority":   "low",
# #         "mode":       "default",
# #     }
# #     # """
# #     # Get nudge for a single user.
# #     # Uses LLM if configured, else rule-based.
# #     # """
# #     # predictions = get_predictions_by_user()
# #     # prediction  = predictions.get(str(user_id))

# #     # if not prediction:
# #     #     return {
# #     #         "user_id":    user_id,
# #     #         "nudge_text": "Welcome! Explore our latest collection.",
# #     #         "offer":      "10% off your first order",
# #     #         "cta":        "Shop Now",
# #     #         "priority":   "low",
# #     #         "mode":       "default",
# #     #     }

# #     if LLM_PROVIDER == "placeholder":
# #         return _rule_nudge(user_id, prediction)
# #     else:
# #         try:
# #             return _llm_nudge(user_id, prediction)
# #         except Exception as e:
# #             print(f"⚠️  LLM nudge failed for user {user_id}: {e}. Using rule fallback.")
# #             return _rule_nudge(user_id, prediction)


# # def generate_all_nudges(use_llm: bool = False) -> list:
# #     """
# #     Pre-generate nudges for ALL users and cache to disk.
# #     Call this once after Module 4 predictions are ready.
# #     """
# #     predictions = load_predictions()
# #     nudges      = []

# #     print(f"🔔 Generating nudges for {len(predictions)} users...")

# #     for i, pred in enumerate(predictions):
# #         uid = pred["user_id"]
# #         if use_llm and LLM_PROVIDER != "placeholder":
# #             nudge = _llm_nudge(uid, pred)
# #         else:
# #             nudge = _rule_nudge(uid, pred)
# #         nudges.append(nudge)

# #         if (i + 1) % 200 == 0:
# #             print(f"   {i+1}/{len(predictions)} done...")

# #     os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
# #     with open(OUT_PATH, "w") as f:
# #         json.dump(nudges, f, indent=2)

# #     # Summary
# #     prio = Counter(n["priority"] for n in nudges)
# #     trans = Counter(
# #         f"{n['current_cluster']}→{n['predicted_cluster']}"
# #         for n in nudges
# #     )
# #     print(f"✅ Nudges saved → {OUT_PATH}")
# #     print(f"   Priority breakdown: {dict(prio)}")
# #     print(f"   Top transitions: {dict(trans.most_common(5))}")
# #     return nudges


# # def load_nudge_cache() -> dict:
# #     """Returns {user_id: nudge_dict} from cached file."""
# #     if not os.path.exists(OUT_PATH):
# #         return {}
# #     with open(OUT_PATH) as f:
# #         nudges = json.load(f)
# #     return {n["user_id"]: n for n in nudges}
