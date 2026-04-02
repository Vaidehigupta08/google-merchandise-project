# 🛍️ Google Merchandise Store — Behavioral AI Recommendation System

An end-to-end AI system that transforms raw user browsing data into **real-time personalized nudges** using behavioral modeling and generative prediction.

---

## 🚀 Pipeline Overview

Raw User Clicks → Behavioral Embeddings → Intent Clusters → Future Prediction → Live Nudges

---

## ⚙️ Modules

### 🔹 Module 1 — Data Cleaning

* Filters noisy user sessions using FFT-based Sincerity Filter
* Converts raw URLs → structured product tokens

### 🔹 Module 2 — Embeddings

* Trains Word2Vec (Beha2Vec) on user journeys
* Converts behavior → dense vectors

### 🔹 Module 3 — Intent Clustering

* Groups users into 5 behavioral clusters
* Learns hidden shopping intents

### 🔹 Module 4 — Prediction (FlowBoost)

* Uses Conditional Flow Matching (CFM)
* Predicts user’s next behavioral state

### 🔹 Module 5 — Agent Interface

* Converts predictions → **Personas + Nudges**
* Exposes FastAPI endpoints
* Includes feedback loop (self-improving system)

---

## 🔥 Key Features

* 🧠 Behavioral AI (not rule-based recommendation)
* 🔮 Future intent prediction (not just past behavior)
* 👤 Auto-generated buyer personas (via LLM)
* 🎯 Real-time personalized nudges
* 🔁 Feedback loop for continuous improvement

---
## 🎨 Frontend (User Experience Layer)

The frontend is a fully interactive e-commerce interface where users can browse products while the system quietly adapts to their behavior in real time.

As users explore the store, the interface continuously adjusts to their actions.
Products, suggestions, and messages are dynamically personalized for each user.
Smart popups appear at the right moment to guide decisions (like offers, urgency, or helpful suggestions).
The system learns from every interaction and improves the experience instantly.

💡 In simple terms:
The website doesn’t just show products — it understands the user and responds intelligently.

✨ What Users Experience
“Recommended for you” products
Real-time smart suggestions
Timely offers and alerts
A personalized shopping journey
⚡ Key Idea

Instead of showing the same store to everyone,
👉 each user gets a unique, adaptive experience based on their behavior.

python -m http.server 5500

## 🛠️ Tech Stack

* Python, PyTorch, NumPy, Pandas
* Gensim (Word2Vec)
* FastAPI + Uvicorn
* Groq / OpenAI (LLM support)

---

## ▶️ How to Run

```bash
# Step 1
python module1/main.py

# Step 2
python module2_beh2vec/beh2vec_train.py

# Step 3
python module3_ELCRec/elcrec_train.py

# Step 4
python module4_Flowboost/main.py

# Step 5 (API)
python module5_agent/m5_main.py --setup
```

---

## 🌐 API

* `GET /nudge/{user_id}` → Get personalized nudge
* `POST /feedback` → Log user interaction
* `GET /docs` → Swagger UI

---

## 📊 Output

* Personas per cluster
* Nudges per user (2500+)
* Predictions per user
* Feedback-based retraining

---

## 💡 Use Case

This system enables e-commerce platforms to:

* Predict user intent before action
* Show proactive UI nudges
* Increase conversions using behavioral AI

---

## 🧠 Author Note

Built as a full-stack AI pipeline combining **ML + DL + Generative AI + Backend systems** into one production-ready project.

---
