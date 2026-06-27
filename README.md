# Spotify Popularity Predictor

End-to-end machine learning project that predicts whether a Spotify track will be popular — and by how much — using its audio features alone.

**Student:** Amila Herenda  
**Course:** University ML Course — Phase 2 Submission  
**Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Maharshi Pandya, Kaggle 2022)

---

## Live App

| Component | URL |
|-----------|-----|
| Streamlit frontend | _TODO: add Streamlit Cloud link after deployment_ |
| FastAPI backend | _TODO: add Render/Railway link after deployment_ |
| API docs (auto-generated) | `<backend-url>/docs` |

---

## What this project does

Given audio features of a song (tempo, energy, danceability, etc.), the app answers two questions:

1. **Classification** — Will this song be popular? (Yes / No, threshold: popularity ≥ 70)
2. **Regression** — What is the predicted popularity score? (0–100)

The app loads pre-trained models from `.pkl` files. It never retrains at runtime.

---

## Architecture

```
OFFLINE (runs once on your machine)
────────────────────────────────────
dataset.csv → train.py → EDA plots + leaderboard + classifier.pkl + regressor.pkl

RUNTIME (live app, always on)
────────────────────────────────────
User → Streamlit (app.py) → HTTP POST → FastAPI (api.py) → loads .pkl → returns prediction
```

See `architecture.mmd` for the full Mermaid diagram.

---

## Project structure

```
spotify-popularity-predictor/
├── data/
│   └── dataset.csv          # download from Kaggle (see below) — not in Git
├── models/
│   ├── classifier.pkl        # saved after running train.py
│   └── regressor.pkl         # saved after running train.py
├── eda_plots/                # PNG charts saved by train.py
├── reports/
│   ├── analysis_report.qmd   # Quarto analysis report (D1)
│   └── slides.qmd            # Quarto reveal.js slides (D4)
├── app/
│   ├── app.py                # Streamlit frontend
│   └── api.py                # FastAPI backend
├── train.py                  # offline training pipeline
├── architecture.mmd          # Mermaid architecture diagram
├── requirements.txt
└── README.md
```

---

## How to reproduce locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/spotify-popularity-predictor.git
cd spotify-popularity-predictor
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the dataset

1. Go to https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
2. Download `dataset.csv`
3. Place it at `data/dataset.csv`

### 4. Run the training pipeline

```bash
python train.py
```

This will:
- Run full EDA and save charts to `eda_plots/`
- Train all models (Dummy → Logistic/Ridge → Decision Tree → XGBoost)
- Print a leaderboard table to the console
- Save `models/classifier.pkl` and `models/regressor.pkl`

### 5. Start the FastAPI backend

```bash
uvicorn app.api:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 6. Start the Streamlit frontend

In a second terminal (with venv activated):

```bash
streamlit run app/app.py
```

App opens at: http://localhost:8501

---

## Models trained

| Task | Baseline | Linear | Decision Tree | XGBoost |
|------|----------|--------|---------------|---------|
| Classification | DummyClassifier | Logistic Regression | DecisionTreeClassifier | XGBClassifier |
| Regression | DummyRegressor | Ridge Regression | DecisionTreeRegressor | XGBRegressor |

All models use `random_state=42`. Preprocessing (scaling, encoding) is fitted on training data only, inside a scikit-learn `Pipeline`, to prevent data leakage.

---

## Key results

Dataset: 106,907 tracks after deduplication | Train: 85,525 | Test: 21,382 | Positive class: 5.1 %

**Classification** (target: popularity ≥ 70)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| DummyClassifier | 0.949 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.579 | 0.082 | 0.711 | 0.147 | 0.712 |
| Decision Tree | 0.491 | 0.083 | 0.895 | 0.152 | 0.722 |
| **XGBoost** | **0.899** | **0.306** | **0.764** | **0.437** | **0.920** |

**Regression** (target: exact popularity score 0–100)

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| DummyRegressor | 0.000 | 17.8 | 21.2 |
| Ridge Regression | 0.045 | 17.0 | 20.8 |
| Decision Tree | 0.092 | 16.2 | 20.2 |
| **XGBoost** | **0.380** | **12.2** | **16.7** |

XGBoost wins both tasks. ROC-AUC of 0.92 means the classifier ranks popular songs well above unpopular ones. R² of 0.38 means audio features explain ~38 % of popularity variance — the rest is artist fame, marketing, and timing, which audio cannot capture.

---

## Reproducibility

- `random_state=42` used in every split, model, and cross-validation call
- All preprocessing steps fitted inside `Pipeline` on training data only
- Exact package versions: run `pip freeze` after installing `requirements.txt`

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| D1 | Reproducible analysis report | `reports/analysis_report.qmd` |
| D2 | Deployed web app | See Live App links above |
| D3 | AI-workflow reflection | `reports/ai_reflection.md` |
| D4 | Presentation slides | `reports/slides.qmd` |
| D5 | Executive summary | `reports/executive_summary.md` |
