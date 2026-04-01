# Module 5 — Agent Interface

Orchestration layer that connects all pipeline outputs to a live API.

## Setup

```bash
cd module5_agent
pip install -r requirements.txt

# Optional: set LLM provider
export LLM_PROVIDER=openai        # or anthropic
export LLM_API_KEY=your_key_here
```

## Run

```bash
# Generate personas + nudges, then start server
python main.py --setup

# Just start server (use cached personas/nudges)
python main.py

# Offline: generate personas only
python main.py --generate-personas

# Offline: generate nudges only
python main.py --generate-nudges
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Interactive API docs (Swagger) |
| GET | `/personas` | All cluster personas |
| GET | `/persona/{cluster_id}` | Single cluster persona |
| POST | `/persona/refresh` | Regenerate personas via LLM |
| GET | `/nudge/{user_id}` | Get nudge for a user |
| POST | `/nudge/generate-all` | Pre-generate all nudges |
| POST | `/feedback` | Log user accept/reject |
| GET | `/feedback/stats` | Cluster acceptance rates |
| POST | `/feedback/trigger-retrain` | Trigger Module 4 retrain |
| GET | `/user/{user_id}` | Full user profile |
| GET | `/cluster/{cluster_id}/users` | Users in a cluster |
| GET | `/clusters` | All cluster summaries |

## Frontend Integration (Next.js)

```js
// 1. Get nudge when user lands on site
const res = await fetch(`http://localhost:8000/nudge/${userId}`)
const nudge = await res.json()
// nudge = { nudge_text, offer, cta, priority, predicted_urls }

// 2. Show popup with nudge.nudge_text + nudge.cta button

// 3. Log feedback when user interacts
await fetch('http://localhost:8000/feedback', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: userId,
    nudge_id: `nudge_${userId}_${Date.now()}`,
    action: 'accepted'   // or 'rejected' or 'ignored'
  })
})
```

## LLM Configuration

| Provider | LLM_PROVIDER value | Package |
|----------|-------------------|---------|
| Placeholder (default) | `placeholder` | none |
| OpenAI GPT-4o | `openai` | `pip install openai` |
| Anthropic Claude | `anthropic` | `pip install anthropic` |

## Output Files

```
module5_agent/outputs/
├── personas.json       ← LLM-generated buyer personas per cluster
├── nudge_cache.json    ← Pre-generated nudges per user
├── reward_log.json     ← All user feedback events
└── reward_weights.json ← Per-user weights for Module 4 retraining
```
