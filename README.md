# Smart Fashion E-Commerce Store & ML Retargeting Bot

An end-to-end **Core Machine Learning** lab project combining **Supervised Purchase Intent Classification** (Random Forest, Gradient Boosting, SVM, Cost-Sensitive Logistic Regression, Naive Bayes) and **Unsupervised Customer Persona Clustering** (K-Means + 2D PCA) with a live **Fashion Storefront** and **Automated WhatsApp & Email Retargeting Bot**.

## Architecture Overview

1. **`frontend/index.html` (Hosted on Vercel):**
   - Full interactive fashion storefront featuring 6 curated outerwear and streetwear products.
   - Customer login modal capturing **Name, Email, and WhatsApp Number**.
   - Real-time behavioral telemetry tracker monitoring **Time on Product (sec)**, **Photos Clicked**, **Size Selections**, **Wishlist/Cart Clicks**, and **Scroll Depth (%)**.
   - **Simulate Tab Close / Exit** trigger (plus real browser `visibilitychange` detection) that sends the live 12-feature session vector to the ML backend.
   - **Live ML Admin & Viva Dashboard** showing the real-time probability gauge, K-Means persona cluster, 5-model cross-validation benchmark chart, feature importance plot, and automated WhatsApp/Email dispatch queue.

2. **`backend/app.py` (Hosted on Render Free Tier):**
   - FastAPI server loading pre-trained ML weights (`backend/ml/model_artifacts.json` & `retargeting_model.pkl`) trained on `fashion_shoppers_sessions.csv` (5,000 sessions modeled after the [UCI Online Shoppers Purchasing Intention Dataset](https://archive-beta.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)).
   - Predicts purchase conversion probability ($0\% - 100\%$) and assigns the shopper to one of three K-Means clusters:
     - **Low Intent (< 30% — Quick Bouncer):** Suppresses message to prevent spam.
     - **Medium Intent (30%–75% — Hesitant Window Shopper):** Sends automated WhatsApp/Email with the exact jacket link + **10% OFF discount code (`JACKET10`)** + K-Means matching outfit recommendation.
     - **High Intent (> 75% — Almost Bought):** Sends automated WhatsApp/Email with a **Reserved Stock & Free Express Delivery (`VIPFREE`)** link without giving away unnecessary margin.

3. **`.github/workflows/keep_alive_cron.yml` (Cron Job Keep-Alive):**
   - Pings `/api/health` and `/api/cron/process-queue` every 10 minutes so your free Render backend stays warm and dispatches queued retargeting messages automatically during your live presentation.

---

## Quick Push to GitHub (`GozonLegendent`)

Open your terminal in the unzipped project folder and run:

```bash
git init
git add .
git commit -m "Initial commit: Smart Fashion Store + Core ML Retargeting Bot"
git branch -M main
git remote add origin https://github.com/GozonLegendent/smart-fashion-ml-retargeting.git
git push -u origin main
```

---

## Free Cloud Deployment (Vercel + Render + Cron Job)

### Step 1: Deploy Backend on Render (100% Free)
1. Go to [Render.com](https://render.com/) -> **New +** -> **Web Service** and connect your GitHub repo.
2. Set **Root Directory** to `backend`, **Build Command** to `pip install -r requirements.txt`, and **Start Command** to `uvicorn app:app --host 0.0.0.0 --port $PORT`.
3. *(Optional for Real Live Emails)* Under **Environment Variables**, add:
   - `SMTP_EMAIL`: Your Gmail address
   - `SMTP_APP_PASSWORD`: Your 16-character Google App Password (from Google Account -> Security -> 2-Step Verification -> App Passwords)
4. Copy your live Render URL (e.g., `https://smart-fashion-ml-retargeting-api.onrender.com`).

### Step 2: Deploy Frontend on Vercel (100% Free)
1. Go to [Vercel.com](https://vercel.com/) -> **Add New Project** -> Import your GitHub repo.
2. Set **Root Directory** to `frontend` and click **Deploy**.
3. Open your live Vercel site, click the **API Settings** badge in the top navbar, and paste your Render Backend URL (if left blank, the frontend automatically runs its built-in browser ML inference engine so your demo never fails even offline!).

### Step 3: Activate the Keep-Alive Cron Job
- **Option A (GitHub Actions - Included):** In your GitHub repo, go to **Settings -> Secrets and variables -> Actions**, add `RENDER_BACKEND_URL` with your Render URL. The included `.github/workflows/keep_alive_cron.yml` pings Render every 10 minutes.
- **Option B (Cron-Job.org - 100% Free):** Create a free timer at [cron-job.org](https://cron-job.org/) pointing to `https://YOUR-RENDER-URL.onrender.com/api/health` every 10 minutes.
