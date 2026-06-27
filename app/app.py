"""
app.py — Streamlit Frontend
Spotify Popularity Predictor
Author: Amila Herenda

Start with:
    streamlit run app/app.py

The app talks to the FastAPI backend (api.py).
Set the backend URL via:
    - Streamlit Cloud secret  :  API_URL = "https://your-app.onrender.com"
    - Environment variable    :  export API_URL="https://your-app.onrender.com"
    - Fallback                :  http://localhost:8000  (local development)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import requests
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================
# WHY read API_URL from an environment variable / secret?
# The backend URL differs between local development (localhost:8000) and
# production (Render/Railway). Hard-coding the URL would break one of the
# two environments. Environment variables let us configure this externally
# without changing any code.
def _get_api_url() -> str:
    try:
        # Streamlit Cloud secrets (set in the app's settings dashboard)
        return st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable, then localhost
        return os.getenv("API_URL", "http://localhost:8000")

API_URL = _get_api_url()

# All 114 genres from the Spotify Tracks Dataset (Maharshi Pandya, Kaggle 2022).
# Displayed as a selectbox so the user picks a known genre rather than typing
# a free-form string. Unknown strings would get OrdinalEncoder code -1.
GENRES = sorted([
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat",
    "british", "cantopop", "chicago-house", "children", "chill", "classical",
    "club", "comedy", "country", "dance", "dancehall", "death-metal",
    "deep-house", "detroit-techno", "disco", "disney", "drum-and-bass", "dub",
    "dubstep", "edm", "electro", "electronic", "emo", "folk", "forro",
    "french", "funk", "garage", "german", "gospel", "goth", "grindcore",
    "groove", "grunge", "guitar", "happy", "hard-rock", "hardcore",
    "hardstyle", "heavy-metal", "hip-hop", "holidays", "honky-tonk", "house",
    "idm", "indian", "indie", "indie-pop", "industrial", "iranian", "j-dance",
    "j-idol", "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino",
    "malay", "mandopop", "metal", "metal-misc", "metalcore", "minimal-techno",
    "movies", "mpb", "new-age", "new-release", "opera", "pagode", "party",
    "philippines-opm", "piano", "pop", "pop-film", "post-dubstep", "power-pop",
    "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b",
    "rainy-day", "reggae", "reggaeton", "road-trip", "rock", "rock-n-roll",
    "rockabilly", "romance", "sad", "salsa", "samba", "sertanejo",
    "show-tunes", "singer-songwriter", "ska", "sleep", "songwriter", "soul",
    "soundtracks", "spanish", "study", "summer", "swedish", "synth-pop",
    "tango", "techno", "trance", "trip-hop", "turkish", "work-out",
    "world-music",
])

PLOTS_DIR = "eda_plots"

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# HEADER
# =============================================================================
st.title("🎵 Spotify Popularity Predictor")
st.markdown(
    "Enter a song's audio features and find out whether it will be **popular** "
    "and what its predicted **popularity score** is."
)

# =============================================================================
# BACKEND HEALTH CHECK
# =============================================================================
# WHY check health on every page load?
# If the FastAPI backend is down (not started, or crashed on Render), we want
# to show a clear error message immediately rather than letting the user fill
# in all sliders and only discovering the problem when they click Predict.
@st.cache_data(ttl=30)   # re-check every 30 seconds, not on every slider move
def _check_backend() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

backend_ok = _check_backend()
if not backend_ok:
    st.error(
        f"**Cannot reach the FastAPI backend** at `{API_URL}`.\n\n"
        "Start it with:\n```\nuvicorn app.api:app --reload --port 8000\n```\n"
        "Then refresh this page."
    )

# =============================================================================
# TABS
# =============================================================================
tab_predict, tab_eda, tab_about = st.tabs(
    ["🎵 Predictor", "📊 EDA Dashboard", "ℹ️ About"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Enter audio features")
    st.caption(
        "Adjust the sliders to match the song you want to evaluate, "
        "then click **Predict**."
    )

    # ── Two-column layout: inputs on the left, results on the right ───────────
    col_inputs, col_results = st.columns([1.2, 1], gap="large")

    with col_inputs:

        # ── Genre & metadata ──────────────────────────────────────────────────
        st.markdown("**Genre & metadata**")
        track_genre = st.selectbox(
            "Genre",
            options=GENRES,
            index=GENRES.index("pop"),
            help="Select the genre closest to your song.",
        )
        c1, c2 = st.columns(2)
        with c1:
            explicit = st.checkbox("Explicit content", value=False)
        with c2:
            duration_min = st.slider(
                "Duration (minutes)", min_value=0.5, max_value=15.0,
                value=3.5, step=0.1,
                help="Track length. 3–4 min is typical for popular songs.",
            )
        duration_ms = int(duration_min * 60_000)

        st.divider()

        # ── Audio features ────────────────────────────────────────────────────
        st.markdown("**Audio features**")
        c3, c4 = st.columns(2)
        with c3:
            danceability = st.slider(
                "Danceability", 0.0, 1.0, 0.65, 0.01,
                help="How suitable for dancing. 0 = not danceable, 1 = very danceable.",
            )
            energy = st.slider(
                "Energy", 0.0, 1.0, 0.60, 0.01,
                help="Intensity and activity. High energy = fast, loud, noisy.",
            )
            valence = st.slider(
                "Valence", 0.0, 1.0, 0.50, 0.01,
                help="Musical positiveness. 1 = happy/euphoric, 0 = sad/angry.",
            )
            acousticness = st.slider(
                "Acousticness", 0.0, 1.0, 0.20, 0.01,
                help="Confidence that the track is acoustic. 1 = fully acoustic.",
            )
        with c4:
            speechiness = st.slider(
                "Speechiness", 0.0, 1.0, 0.05, 0.01,
                help="Presence of spoken words. >0.66 = podcast/audiobook.",
            )
            instrumentalness = st.slider(
                "Instrumentalness", 0.0, 1.0, 0.00, 0.01,
                help="Likelihood of no vocals. >0.5 = probably instrumental.",
            )
            liveness = st.slider(
                "Liveness", 0.0, 1.0, 0.10, 0.01,
                help="Presence of an audience. >0.8 = probably a live recording.",
            )
            loudness = st.slider(
                "Loudness (dB)", -60.0, 5.0, -7.0, 0.5,
                help="Overall loudness. Typical studio tracks: -10 to -5 dB.",
            )

        st.divider()

        # ── Musical structure ─────────────────────────────────────────────────
        st.markdown("**Musical structure**")
        c5, c6, c7 = st.columns(3)
        with c5:
            tempo = st.slider(
                "Tempo (BPM)", 30.0, 250.0, 120.0, 1.0,
                help="Beats per minute. 120 BPM = standard dance track.",
            )
        with c6:
            key = st.selectbox(
                "Key",
                options=list(range(12)),
                format_func=lambda k: ["C","C#","D","D#","E","F",
                                       "F#","G","G#","A","A#","B"][k],
                index=5,
                help="Musical key. 0=C, 1=C#, … 11=B.",
            )
        with c7:
            mode = st.radio(
                "Mode", options=[1, 0],
                format_func=lambda m: "Major" if m == 1 else "Minor",
                index=0,
                help="Major sounds brighter; minor sounds darker.",
            )
        time_signature = st.select_slider(
            "Time signature", options=[3, 4, 5, 6, 7], value=4,
            help="Beats per bar. 4/4 is by far the most common in popular music.",
        )

        st.divider()

        # ── Predict button ────────────────────────────────────────────────────
        predict_btn = st.button(
            "Predict popularity", type="primary", use_container_width=True,
            disabled=not backend_ok,
        )

    # ── Right column: results ─────────────────────────────────────────────────
    with col_results:
        st.markdown("**Prediction result**")

        if predict_btn:
            payload = {
                "danceability":    danceability,
                "energy":          energy,
                "loudness":        loudness,
                "speechiness":     speechiness,
                "acousticness":    acousticness,
                "instrumentalness": instrumentalness,
                "liveness":        liveness,
                "valence":         valence,
                "tempo":           tempo,
                "key":             int(key),
                "mode":            int(mode),
                "time_signature":  int(time_signature),
                "duration_ms":     duration_ms,
                "explicit":        int(explicit),
                "track_genre":     track_genre,
            }

            with st.spinner("Asking the model …"):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                except requests.exceptions.ConnectionError:
                    st.error("Backend not reachable. Is `uvicorn app.api:app` running?")
                    result = None

                except requests.exceptions.HTTPError as e:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    result = None

            if result:
                popular      = result["popular"]
                score        = result["popularity_score"]
                confidence   = result["confidence"]

                # ── Verdict banner ────────────────────────────────────────────
                if popular:
                    st.success("### This song is likely POPULAR")
                else:
                    st.warning("### This song is likely NOT popular")

                # ── Key metrics ───────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted score", f"{score:.1f} / 100")
                m2.metric("Confidence",       f"{confidence * 100:.1f} %")
                m3.metric("Threshold",         "70 / 100")

                # ── Popularity bar ────────────────────────────────────────────
                st.markdown("**Score on the 0–100 scale:**")
                st.progress(int(score))

                # ── Confidence note ───────────────────────────────────────────
                st.caption(
                    f"The classifier gives a **{confidence*100:.1f} % probability** "
                    f"that this song is popular (popularity ≥ 70). "
                    f"The regressor predicts a score of **{score:.1f}**."
                )

                # ── Feature summary ───────────────────────────────────────────
                with st.expander("Show the features you submitted"):
                    import pandas as pd
                    feature_df = pd.DataFrame([payload]).T
                    feature_df.columns = ["Value"]
                    st.dataframe(feature_df, use_container_width=True)

        else:
            # Placeholder before the user clicks Predict
            st.info(
                "Adjust the sliders on the left to describe your song, "
                "then click **Predict popularity**."
            )
            st.markdown("""
**Quick guide — what the audio features mean:**

| Feature | Low value | High value |
|---------|-----------|------------|
| Danceability | Hard to dance to | Easy to dance to |
| Energy | Calm, soft | Intense, loud |
| Valence | Sad / angry | Happy / euphoric |
| Acousticness | Electronic / synthetic | Acoustic instruments |
| Speechiness | Pure music | Mostly spoken words |
| Instrumentalness | Has vocals | No vocals |
| Liveness | Studio recording | Live audience |
""")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.subheader("Exploratory Data Analysis — saved charts")
    st.caption(
        "These charts were generated by `train.py` from the raw Spotify dataset. "
        "They show what the data looks like before any modelling."
    )

    # Helper: show a plot if the file exists; otherwise show a placeholder
    def _show_plot(filename: str, caption: str) -> None:
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(
                f"**{filename}** not found.  \n"
                "Run `python train.py` to generate EDA plots."
            )

    # ── Section 1: Target variable ────────────────────────────────────────────
    st.markdown("### Target variable — Popularity")
    _show_plot(
        "01_popularity_distribution.png",
        "Left: raw distribution of popularity scores (0–100). "
        "Right: class balance — how many songs cross the 70 threshold.",
    )

    # ── Section 2: Feature distributions ─────────────────────────────────────
    st.markdown("### Audio feature distributions")
    _show_plot(
        "02_feature_distributions.png",
        "Histogram of each numeric feature. "
        "Helps spot skewed or bimodal distributions.",
    )

    # ── Section 3: Correlations ───────────────────────────────────────────────
    st.markdown("### Correlations")
    _show_plot(
        "03_correlation_heatmap.png",
        "Pearson correlation matrix (lower triangle). "
        "Red = positive correlation, blue = negative. "
        "Look at the bottom row for features most correlated with popularity.",
    )

    # ── Section 4: Outliers ────────────────────────────────────────────────────
    st.markdown("### Outlier detection")
    _show_plot(
        "04_outlier_boxplots.png",
        "Boxplots for key features. Dots beyond the whiskers are outliers. "
        "RobustScaler handles these without explicit removal.",
    )

    # ── Section 5: Feature–target relationships ───────────────────────────────
    st.markdown("### Feature–target relationships")
    _show_plot(
        "05_feature_target_scatter.png",
        "Each audio feature plotted against the raw popularity score. "
        "A positive slope means higher values tend to be more popular.",
    )

    # ── Section 6: Genre popularity ───────────────────────────────────────────
    st.markdown("### Genre analysis")
    _show_plot(
        "06_genre_popularity.png",
        "Top 25 genres by mean popularity. "
        "Genre is a useful feature — some genres are systematically more popular.",
    )

    # ── Section 7: Model evaluation plots ─────────────────────────────────────
    st.markdown("### Model evaluation (XGBoost — best model)")
    c_left, c_right = st.columns(2)
    with c_left:
        _show_plot(
            "07_confusion_matrix_xgb.png",
            "Confusion matrix on the held-out test set. "
            "True positives = popular songs correctly identified.",
        )
        _show_plot(
            "08_roc_curve_xgb.png",
            "ROC curve: trade-off between true positive rate and false positive rate. "
            "AUC closer to 1.0 = better classifier.",
        )
    with c_right:
        _show_plot(
            "09_predicted_vs_actual_xgb.png",
            "Predicted vs actual popularity scores. "
            "Perfect model = all points on the red diagonal.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
**Student:** Amila Herenda
**Course:** University ML Course
**Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
(Maharshi Pandya, Kaggle 2022) — ~114 000 tracks, 21 audio features, 114 genres.

---

### Architecture

```
OFFLINE  (runs once on your machine)
──────────────────────────────────────────────────────────
dataset.csv → train.py → EDA plots + classifier.pkl + regressor.pkl

RUNTIME  (this app)
──────────────────────────────────────────────────────────
You → Streamlit (this page) → HTTP POST → FastAPI → .pkl → prediction
```

### Two ML tasks

| Task | Target | Best model |
|------|--------|------------|
| Classification | popularity ≥ 70 → 1, else 0 | XGBoost (tuned) |
| Regression | exact popularity score 0–100 | XGBoost (tuned) |

### Model ladder trained

1. **DummyClassifier / DummyRegressor** — always-predict-mean baseline
2. **Logistic Regression / Ridge** — linear models
3. **Decision Tree** — non-linear, interpretable
4. **XGBoost** — gradient boosting with RandomizedSearchCV

### Key design decisions

- `random_state=42` everywhere → fully reproducible results
- scikit-learn `Pipeline` → no data leakage (scaler fitted on training data only)
- `RobustScaler` instead of `StandardScaler` → handles outliers in loudness / tempo
- Models saved as `.pkl` → backend loads once at startup, never retrains

---
**Backend URL:** `{API_URL}`
**API docs:** [{API_URL}/docs]({API_URL}/docs)
""")
