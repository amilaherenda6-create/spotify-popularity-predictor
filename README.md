# 🎵 Spotify Popularity Predictor

> End-to-end machine learning project — predicts whether a song will be popular on Spotify and estimates its exact popularity score, using audio features alone.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-1DB954?style=for-the-badge&logo=streamlit)](https://spotify-popularity-predictor-dkb249nsp7axa3hkxdpkht.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-0066FF?style=for-the-badge&logo=render)](https://spotify-popularity-predictor-5bmz.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/amilaherenda6/spotify-popularity-predictor)

---

| | |
|---|---|
| **Student** | Herend Amila |
| **Course** | Modelling in Advanced Data Analytics — FELU |
| **Professors** | <span style="color:#0F62FE">**Uroš Godnov**</span> · <span style="color:#0F62FE">**Aleš Gorišek**</span> |
| **Date** | 28.06.2026 |
| **Dataset** | [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — Maharshi Pandya, Kaggle 2022 |

---

## 🚀 Live Deployment

| Component | URL |
|-----------|-----|
| **Streamlit app** (frontend) | https://spotify-popularity-predictor-dkb249nsp7axa3hkxdpkht.streamlit.app |
| **FastAPI backend** | https://spotify-popularity-predictor-5bmz.onrender.com |
| **API docs** (Swagger UI) | https://spotify-popularity-predictor-5bmz.onrender.com/docs |

> **Note:** The Render backend runs on the free tier and may take ~30 seconds to wake up after inactivity. Subsequent requests are instant.

---

## What This Project Does

Given 15 audio features of any song (tempo, energy, danceability, loudness, etc.), the app answers two questions simultaneously:

| Task | Question | Method |
|------|----------|--------|
| **Classification** | Will this song be popular? (Yes / No) | popularity ≥ 70 → label 1 |
| **Regression** | What is the predicted popularity score? | predicts 0–100 directly |

The app loads pre-trained models from `.pkl` files at startup. **It never retrains at runtime.**

---

## Architecture

```
OFFLINE — Training Phase  (runs once on your machine)
──────────────────────────────────────────────────────────────
dataset.csv
    → EDA (6 charts saved to eda_plots/)
    → Data cleaning (drop duplicates, encode genres, drop ID columns)
    → Feature engineering (5 new features: ratios, interactions)
    → Stratified 80/20 train/test split  (random_state=42)
    → scikit-learn Pipeline (RobustScaler + OrdinalEncoder — no leakage)
    → Model ladder: Dummy → Logistic/Ridge → Decision Tree → XGBoost + CV
    → Leaderboard evaluation on held-out test set
    → classifier.pkl + regressor.pkl saved to models/

RUNTIME — Inference Phase  (live app, always on)
──────────────────────────────────────────────────────────────
User (browser)
    → Streamlit app.py  (sliders for all audio features)
    → HTTP POST /predict  (JSON payload)
    → FastAPI api.py  (loads .pkl once at startup)
    → Pipeline.predict()  (preprocessing + model in one call)
    → JSON response  →  Streamlit displays result
```

See [`architecture.mmd`](architecture.mmd) for the full Mermaid diagram.

---

## Project Structure

```
spotify-popularity-predictor/
├── app/
│   ├── api.py                    # FastAPI backend — loads .pkl, serves /predict
│   └── app.py                    # Streamlit frontend — sliders + results + EDA dashboard
├── data/
│   └── dataset.csv               # download from Kaggle (not in Git — 20 MB)
├── eda_plots/                    # 9 PNG charts saved by train.py
├── models/
│   ├── classifier.pkl            # fitted XGBoost Pipeline (0.73 MB)
│   └── regressor.pkl             # fitted XGBoost Pipeline (0.93 MB)
├── reports/
│   ├── analysis_report.qmd       # D1 — Quarto analysis report
│   ├── slides.qmd                # D4 — reveal.js slide deck
│   ├── ai_workflow_reflection.md # D3 — AI workflow reflection
│   └── executive_summary.md     # D5 — executive summary
├── .streamlit/
│   └── config.toml               # Streamlit theme (Spotify green)
├── architecture.mmd              # Mermaid architecture diagram
├── render.yaml                   # Render deployment config
├── requirements.txt              # all Python dependencies
├── train.py                      # offline training pipeline (935 lines)
└── README.md
```

---

## How to Reproduce Locally

### 1 — Clone

```bash
git clone https://github.com/amilaherenda6/spotify-popularity-predictor.git
cd spotify-popularity-predictor
```

### 2 — Install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Download the dataset

1. Go to https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Download `dataset.csv`
3. Place it at `data/dataset.csv`

> The `.pkl` model files are already in the repository — you can skip training and go straight to step 5 if you just want to run the app.

### 4 — Run the training pipeline (optional — models already in repo)

```bash
python train.py
```

This will:
- Run full EDA and save 9 charts to `eda_plots/`
- Train all 8 models (4 classifiers + 4 regressors)
- Print the leaderboard to the console
- Overwrite `models/classifier.pkl` and `models/regressor.pkl`

Estimated runtime: **8–12 minutes** (XGBoost cross-validation is the slow step).

### 5 — Start the FastAPI backend

```bash
# If uvicorn is on your PATH:
uvicorn app.api:app --reload --port 8000

# If not (Windows common issue):
python -m uvicorn app.api:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### 6 — Start the Streamlit frontend

Open a second terminal (with venv activated):

```bash
python -m streamlit run app/app.py
```

App opens automatically at: http://localhost:8501

---

## Key Results

**Dataset:** 106,907 tracks (after deduplication) | Train: 85,525 | Test: 21,382 | `random_state=42`

**Classification** — target: popularity ≥ 70 → popular (only 5.1% of songs qualify)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| DummyClassifier (baseline) | 0.949 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.579 | 0.082 | 0.711 | 0.147 | 0.712 |
| Decision Tree (depth=6) | 0.491 | 0.083 | 0.895 | 0.152 | 0.722 |
| **XGBoost** ✓ | **0.899** | **0.306** | **0.764** | **0.437** | **0.920** |

**Regression** — target: exact popularity score (0–100)

| Model | R² | MAE | RMSE |
|-------|:--:|:---:|:----:|
| DummyRegressor (baseline) | 0.000 | 17.8 | 21.2 |
| Ridge Regression | 0.045 | 17.0 | 20.8 |
| Decision Tree (depth=6) | 0.092 | 16.2 | 20.2 |
| **XGBoost** ✓ | **0.380** | **12.2** | **16.7** |

XGBoost wins both tasks. ROC-AUC of **0.92** shows the classifier reliably separates popular from unpopular songs. R² of **0.38** means audio features explain roughly 38% of popularity variance — the remaining 62% is driven by artist fame, marketing, release timing, and social media, which audio cannot capture.

---

## Deliverables

| # | Deliverable | File |
|---|-------------|------|
| D1 | Reproducible analysis report (Quarto) | [`reports/analysis_report.qmd`](reports/analysis_report.qmd) |
| D2 | Deployed web app (Streamlit + FastAPI) | [Live app](https://spotify-popularity-predictor-dkb249nsp7axa3hkxdpkht.streamlit.app) |
| D3 | AI workflow reflection | [`reports/ai_workflow_reflection.md`](reports/ai_workflow_reflection.md) |
| D4 | Presentation slides (Quarto reveal.js) | [`reports/slides.qmd`](reports/slides.qmd) |
| D5 | Executive summary (plain language) | [`reports/executive_summary.md`](reports/executive_summary.md) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| ML pipeline | scikit-learn (Pipeline, ColumnTransformer, RobustScaler, OrdinalEncoder, SelectKBest) |
| Models | DummyClassifier/Regressor, LogisticRegression, Ridge, DecisionTree, XGBoost |
| Model persistence | joblib (.pkl files) |
| Backend API | FastAPI + uvicorn + Pydantic |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud (frontend) + Render (backend) |
| Version control | Git + GitHub |

---

## Reproducibility

- `random_state=42` used in every `train_test_split`, model constructor, and `RandomizedSearchCV` call
- All preprocessing (scaling, encoding, imputation) fitted inside a scikit-learn `Pipeline` — **no data leakage**
- `RobustScaler` used instead of `StandardScaler` — robust to the outliers present in `loudness`, `tempo`, and `duration_ms`
- Models committed to the repository — results are reproducible without re-running `train.py`
