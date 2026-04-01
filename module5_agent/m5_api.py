
import json
import os
import sys

# FIX: ensure root path bhi include ho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from module5_agent.m5_data_loader import (
    load_cluster_map,
    load_intents,
    load_predictions,
    get_cluster_url_stats,
    get_cluster_sizes,
    invalidate_cache,
)
from module5_agent.m5_persona_engine import (
    generate_all_personas,
    get_persona_for_cluster,
    load_personas,
)
from module5_agent.m5_nudge_engine import (
    get_nudge_for_user,
    generate_all_nudges,
    load_nudge_cache,
)
from module5_agent.m5_feedback_loop import (
    log_feedback,
    get_cluster_reward_stats,
    trigger_retrain,
    compute_reward_weights,
)


# ── App setup ──────────────────────────────────────────

app = FastAPI(
    title="Module 5 — Agent Interface",
    description="Orchestration layer: personas, nudges, feedback loop",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id: str
    nudge_id: str
    action: str
    cluster_id: Optional[int] = None
    predicted_cluster: Optional[int] = None


class NudgeGenerateRequest(BaseModel):
    use_llm: bool = Field(default=False)


class PersonaRefreshRequest(BaseModel):
    force: bool = Field(default=False)


# ── Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "docs": "/docs"}


@app.get("/nudge/{user_id}")
def get_nudge(user_id: str):
    return get_nudge_for_user(user_id)


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    try:
        event = log_feedback(
            user_id=req.user_id,
            nudge_id=req.nudge_id,
            action=req.action,
            cluster_id=req.cluster_id,
            predicted_cluster=req.predicted_cluster,
        )
        return {"status": "logged", "event": event}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/feedback/stats")
def feedback_stats():
    return get_cluster_reward_stats()


@app.post("/feedback/trigger-retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(trigger_retrain)
    return {"status": "retrain triggered"}


# ── Persona routes ─────────────────────────────────────────────────────

@app.get("/personas")
def get_all_personas():
    return load_personas()


@app.get("/persona/{cluster_id}")
def get_persona(cluster_id: int):
    persona = get_persona_for_cluster(cluster_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"No persona for cluster {cluster_id}")
    return persona


@app.post("/persona/refresh")
def refresh_personas(req: PersonaRefreshRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_all_personas, req.force)
    return {"status": "persona refresh started", "force": req.force}


# ── Nudge routes ───────────────────────────────────────────────────────

@app.post("/nudge/generate-all")
def generate_nudges(req: NudgeGenerateRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(generate_all_nudges, req.use_llm)
    return {"status": "nudge generation started", "use_llm": req.use_llm}


# ── User & Cluster info routes ─────────────────────────────────────────

@app.get("/user/{user_id}")
def get_user_profile(user_id: str):
    cmap        = load_cluster_map()
    cluster_id  = cmap.get(str(user_id))
    nudge       = get_nudge_for_user(user_id)
    persona     = get_persona_for_cluster(cluster_id) if cluster_id is not None else None

    return {
        "user_id":    user_id,
        "cluster_id": cluster_id,
        "persona":    persona,
        "nudge":      nudge,
    }


@app.get("/cluster/{cluster_id}/users")
def get_cluster_users(cluster_id: int):
    cmap  = load_cluster_map()
    users = [uid for uid, cid in cmap.items() if cid == cluster_id]
    return {"cluster_id": cluster_id, "user_count": len(users), "user_ids": users}


@app.get("/clusters")
def get_all_clusters():
    intents = load_intents()
    sizes   = get_cluster_sizes()
    result  = []
    for cid_str, info in intents.items():
        cid     = int(cid_str)
        persona = get_persona_for_cluster(cid)
        result.append({
            "cluster_id":   cid,
            "size":         sizes.get(cid, 0),
            "persona_name": persona.get("persona_name") if persona else None,
            "top_interests": persona.get("top_interests") if persona else [],
        })
    return sorted(result, key=lambda x: x["cluster_id"])
# import json
# import os
# import sys

# # FIX: ensure root path bhi include ho
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(BASE_DIR)
# sys.path.insert(0, PARENT_DIR)

# from fastapi import FastAPI, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import Optional

# from module5_agent.m5_data_loader import (
#     load_cluster_map,
#     load_intents,
#     load_predictions,
#     get_cluster_url_stats,
#     get_cluster_sizes,
#     invalidate_cache,
# )
# from module5_agent.m5_persona_engine import (
#     generate_all_personas,
#     get_persona_for_cluster,
#     load_personas,
# )
# from module5_agent.m5_nudge_engine import (
#     get_nudge_for_user,
#     generate_all_nudges,
#     load_nudge_cache,
# )
# from module5_agent.m5_feedback_loop import (
#     log_feedback,
#     get_cluster_reward_stats,
#     trigger_retrain,
#     compute_reward_weights,
# )


# # ── App setup ──────────────────────────────────────────

# app = FastAPI(
#     title="Module 5 — Agent Interface",
#     description="Orchestration layer: personas, nudges, feedback loop",
#     version="1.0.0",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ── Models ─────────────────────────────────────────────

# class FeedbackRequest(BaseModel):
#     user_id: str
#     nudge_id: str
#     action: str
#     cluster_id: Optional[int] = None
#     predicted_cluster: Optional[int] = None


# class NudgeGenerateRequest(BaseModel):
#     use_llm: bool = Field(default=False)


# class PersonaRefreshRequest(BaseModel):
#     force: bool = Field(default=False)


# # ── Routes ─────────────────────────────────────────────

# @app.get("/")
# def root():
#     return {"status": "running", "docs": "/docs"}


# @app.get("/nudge/{user_id}")
# def get_nudge(user_id: str):
#     return get_nudge_for_user(user_id)


# @app.post("/feedback")
# def submit_feedback(req: FeedbackRequest):
#     try:
#         event = log_feedback(
#             user_id=req.user_id,
#             nudge_id=req.nudge_id,
#             action=req.action,
#             cluster_id=req.cluster_id,
#             predicted_cluster=req.predicted_cluster,
#         )
#         return {"status": "logged", "event": event}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))