import os
import json


# ── Config (same hi rakha) ────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")   # default groq kar diya
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "")
LLM_MODEL    = os.getenv("LLM_MODEL",    "")  # groq model


# ── Unified call interface (same structure) ───────────────────

def llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """
    Single function to call Groq LLM.
    """
    return _call_groq(system_prompt, user_prompt, max_tokens)


# ── Groq implementation ───────────────────────────────────────

def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Install groq: pip install groq")

    client = Groq(api_key=LLM_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# ── Placeholder (same rakha for safety) ───────────────────────

def _call_placeholder(system_prompt: str, user_prompt: str) -> str:
    cluster_id = "unknown"
    for word in user_prompt.split():
        if word.isdigit():
            cluster_id = word
            break

    return json.dumps({
        "persona_name":    f"Cluster {cluster_id} Shopper",
        "age_range":       "25-40",
        "shopping_style":  "Placeholder — connect LLM API to generate real persona",
        "top_interests":   ["Accessories", "Clothing", "Bags"],
        "purchase_intent": "medium",
        "nudge_strategy":  "Highlight trending items and limited-time offers",
    }, indent=2)