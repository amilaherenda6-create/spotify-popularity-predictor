# D3 — AI Workflow Reflection

**Student:** Herend Amila
**Date:** 28.06.2026
**Course:** Modelling in Advanced Data Analytics — FELU
**Professors:** <span style="color:#0F62FE">**Uroš Godnov**</span> · <span style="color:#0F62FE">**Aleš Gorišek**</span>
**Project:** Spotify Popularity Predictor

---

## Overview

This reflection describes how I used AI-assisted development tools throughout the Spotify Popularity Predictor project, how I verified the AI output, what went wrong, and what I learned.

---

## Tools Used

### Primary tool — Claude Code (Anthropic)

Claude Code is a command-line AI assistant that works directly inside VS Code. It can read, write, and edit files in the project folder, run terminal commands, and explain its reasoning at each step.

I used Claude Code for the following tasks:

| Task | What Claude Code did |
|------|----------------------|
| Project scaffolding | Created the full folder structure (`data/`, `models/`, `eda_plots/`, `reports/`, `app/`) |
| `requirements.txt` | Generated all package dependencies with inline explanations |
| `train.py` | Wrote the full 935-line offline training pipeline |
| `app/api.py` | Wrote the FastAPI backend with lifespan model loading and Pydantic validation |
| `app/app.py` | Wrote the Streamlit frontend with two tabs and health checking |
| `architecture.mmd` | Generated the Mermaid diagram source file |
| Deployment | Created `render.yaml` and `.streamlit/config.toml`, fixed `.gitignore`, staged and pushed commits |
| Debugging | Diagnosed and fixed the `Unnamed: 0` leakage issue and the Windows Unicode encoding errors |

### Secondary tool — Claude.ai (web interface)

Used for conceptual questions during the planning phase — for example, clarifying the difference between `StandardScaler` and `RobustScaler`, and understanding why `stratify=` matters in imbalanced classification.

### MCP servers used within Claude Code

- **context7** — fetched up-to-date documentation for FastAPI (lifespan API, Pydantic v2 `model_dump()`), scikit-learn (`ColumnTransformer`, `SelectKBest`), and XGBoost (`scale_pos_weight`). This was important because my training data (August 2025) may not reflect the latest API changes.
- **Playwright** — attempted browser automation to screenshot the running Streamlit app; blocked by localhost sandbox restrictions.
- **Mermaid** — attempted live diagram preview; blocked by missing Puppeteer/Chrome install on my machine. Diagram was saved as `.mmd` text file instead.

---

## How I Verified the AI Output

I did not blindly accept any file Claude Code produced. My verification process for each file:

**`train.py`**
- Read the entire file in VS Code before running it
- Ran `python train.py` myself and watched the console output step by step
- Confirmed: 6 EDA plots saved, 8 models trained, leaderboard printed, 2 `.pkl` files created
- Checked that `random_state=42` appeared in every relevant call (I searched for it with Ctrl+F)
- Verified that the Pipeline structure prevented data leakage by confirming no scaler was fitted before `pipeline.fit(X_train, ...)`

**`app/api.py`**
- Opened `http://localhost:8000/docs` and tested the `/predict` endpoint manually with different inputs
- Confirmed the response JSON matched the `PredictionResponse` Pydantic schema

**`app/app.py`**
- Opened `http://localhost:8501` in the browser
- Tested the Predictor tab with a typical pop song (high danceability + energy → predicted popular)
- Tested with a classical instrumental (low energy + high instrumentalness → predicted not popular)
- Verified the EDA Dashboard tab loaded all 9 charts

**Architecture diagram**
- Read `architecture.mmd` and traced every arrow manually to confirm it matched the actual code flow

---

## What Went Wrong — and How We Fixed It

### Issue 1 — Windows Unicode encoding error

When `train.py` ran for the first time, it crashed immediately with:

```
UnicodeEncodeError: 'charmap' codec can't encode character '→'
```

**Cause:** The Windows terminal uses `cp1252` encoding by default, which cannot display the `→` arrow character I had in print statements.

**Fix:** Claude Code ran a Python script to replace all non-ASCII characters (`→`, `…`, `–`, `²`) with ASCII equivalents (`->`, `...`, `-`, `2`) throughout `train.py`. We also set `PYTHONIOENCODING=utf-8` as an environment variable for subsequent runs.

**What I learned:** Windows terminal encoding is a real-world problem that production code needs to handle. ASCII-safe print statements are more portable.

### Issue 2 — `Unnamed: 0` column treated as a feature

In the first training run, the CSV's row-index column (`Unnamed: 0`) was included as a numeric feature. This inflated the model performance slightly (R² = 0.46 in first run vs 0.38 after fix) because row order had a spurious correlation with popularity in the original dataset ordering.

**Fix:** Added `"Unnamed: 0"` to the list of identifier columns dropped in `clean_data()`. This also revealed 7,093 genuine duplicate rows that had been hidden by the different row indices.

**What I learned:** Always inspect what columns your preprocessor actually sees. The `Numeric features (20): [...]` print statement Claude Code added to `build_preprocessor()` made this bug immediately visible.

### Issue 3 — `uvicorn` command not found on Windows

Running `uvicorn app.api:app` in the terminal failed because the Scripts folder was not on the PATH.

**Fix:** Used `python -m uvicorn app.api:app` instead — this runs uvicorn as a Python module, which always works regardless of PATH configuration.

### Issue 4 — Mermaid live preview failed

The `mermaid-diagrams` MCP tool requires Puppeteer (headless Chrome) to render diagrams. This was not installed.

**Fix:** Saved the `.mmd` source text file directly. The diagram can be viewed by pasting into [mermaid.live](https://mermaid.live). The text format is actually what my professor asked for anyway.

---

## Effort and Time

| Phase | Estimated time |
|-------|---------------|
| Project planning and scaffolding | 30 minutes |
| Writing and reviewing `train.py` | 45 minutes |
| Writing and reviewing `api.py` + `app.py` | 30 minutes |
| Running `train.py` and debugging | 25 minutes |
| Testing the live app locally | 15 minutes |
| Deployment (Render + Streamlit Cloud) | 20 minutes |
| Writing deliverable documents | 60 minutes |
| **Total** | **~3.5 hours** |

Without AI assistance, writing `train.py` alone (935 lines covering EDA, cleaning, feature engineering, model ladder, leaderboard, and model saving) would have taken me at least 2–3 days. AI assistance compressed this to reviewing and verifying rather than authoring from scratch.

---

## What I Learned About AI-Assisted Development

**1. AI generates fast but you must verify slowly.**
The code Claude Code produced was largely correct, but it did not know about Windows encoding issues or the `Unnamed: 0` artifact in my specific CSV. These required me to read the output carefully and catch the problems myself.

**2. Explanation-first prompting produces better code.**
Because I asked Claude Code to explain every decision (for my professor's questions), the resulting comments in `train.py` actually helped me understand why `RobustScaler` was chosen over `StandardScaler`, why the Pipeline prevents leakage, and why `stratify=` is needed. I can now defend every line.

**3. The AI is a senior pair programmer, not a replacement for understanding.**
I still needed to understand what `scale_pos_weight` does in XGBoost, what `ROC-AUC` measures, and why an R² of 0.38 is an honest result rather than a failure. The AI produced the code; I had to provide the domain judgment about whether the results made sense.

**4. MCP servers for live documentation are genuinely useful.**
Fetching the FastAPI lifespan documentation via context7 ensured the code used the current API pattern (`@asynccontextmanager`) rather than the deprecated `@app.on_event("startup")` pattern from older tutorials.
