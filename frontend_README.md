# Frontend UI - Google Merchandise Store (AI Personalized)

This is the main customer-facing frontend for the AI-Personalized E-commerce Project. It is built entirely in **Vanilla HTML, CSS, and JavaScript** (Zero external frameworks like React/Vue) to ensure maximum speed and simple integration.

## 🚀 Quick Start
1. Ensure your Module 5 Backend is running: `uvicorn module5_agent.m5_api:app --host 0.0.0.0 --port 8000`
2. Open `frontend.html` directly in any web browser (Chrome/Edge/Firefox). No build step or Live Server strictly required (though Live Server is recommended for avoiding CORS issues).

---

## ✨ Key Features

### 1. Real-Time AI Personalization (Module 5 Integration)
The frontend communicates heavily with the FastAPI backend (`http://localhost:8000`) to fetch:
- **Buyer Persona:** e.g., "The Tech Trendsetter"
- **Cluster ID:** Maps the user to a specific dynamic behavioral cluster.
- **Nudges:** Receives push-style popups (Discount, Urgency, Recommendation) based on predictive models.

### 2. Event-Driven Feedback Loop
Every action the user takes is recorded locally and synced via the API:
- `product_view`
- `cart_add`
- `wishlist_add`
- `nudge_accepted` / `nudge_ignored`
These events are sent to the `/feedback` endpoint to dynamically retrain the reinforcement learning models in the backend.

### 3. Dynamic Fallback Catalog
- Starts with **8 Core Google Merch Products**.
- Automatically fetches 200+ expanded items asynchronously using `DummyJSON` and `FakeStoreAPI` to populate a rich simulation catalog.

### 4. Local State Management
Uses HTML5 `localStorage` for:
- Mock User Authentication & Session (`gm_users`, `gm_active_user`)
- Cart State (`gm_cart_{user_id}`)
- Event Logs (`gm_events_{user_id}`)

### 5. Context-Aware Fallbacks
Since the offline prediction models currently output static assignments (e.g., `Electronics` for User 118), the frontend contains intelligent string sanitization. It dynamically overwrites the incoming static text with the user's **currently active browsing category** on the fly so popups feel highly relevant to the user's immediate clicks.

---

## 📂 Page Structure

The application is a Single Page Application (SPA), routing through JS `goPage()` state toggles:
- **Home (`#page-home`):** Hero section, AI Live Status Banner, top categories.
- **Shop (`#page-shop`):** Product grid, Urgency bars, AI recommendation stripes, category filtering.
- **Detail (`#page-detail`):** Deep dive into a product, "Why AI picked this for you" reasoning, Add to Cart logic.
- **Cart (`#page-cart`):** Cart summary, dynamic AI cross-sell recommendations ("Because you added X, buy Y").
- **Wishlist (`#page-wishlist`):** Saved items.
- **About AI (`#page-about-ai`):** An educational view with an interactive **Live API Tester** to simulate specific `user_id` payloads.

---

## 🛠️ Modifying the Code
- **UI Customization:** Look for the `<style>` block at the top. Variables are defined in `:root`.
- **API Endpoints:** The base URL is located at line `1203`: `const API='http://localhost:8000';`.
- **Nudge Logic:** Handled in `refreshNudgeAfterAction()` and `showNudge()` functions. String masking rules exist here to format raw Model predictions.