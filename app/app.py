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
# CUSTOM CSS — dark premium theme
# =============================================================================
st.markdown("""
<style>
/* ── Base & background ─────────────────────────────────────────────────── */
.stApp {
    background-color: #0A0A0F;
    color: #E0E0E0;
}
.stApp > header {
    background-color: transparent;
}
[data-testid="stAppViewContainer"] {
    background-color: #0A0A0F;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stToolbar"] {
    right: 1rem;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0d0d18;
    border-right: 1px solid #1a1a2e;
}

/* ── Remove default white blocks ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-panel"] {
    background: transparent;
    padding: 0;
}
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stHorizontalBlock"]) {
    background: transparent;
}

/* ── Typography ────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: #F0F0F0 !important;
}
p, li, label {
    color: #C0C0C0;
}
.stMarkdown p {
    color: #C0C0C0;
}

/* ── Tabs — pill style ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #111118;
    border-radius: 14px;
    padding: 5px 6px;
    border: 1px solid #1e1e2e;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #888;
    padding: 8px 22px;
    font-weight: 500;
    font-size: 0.9em;
    background: transparent;
    border: none;
    transition: all 0.2s ease;
    letter-spacing: 0.3px;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #fff;
    background: #1a1a28;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1DB954, #17a349) !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(29, 185, 84, 0.35);
}

/* ── Primary button — large green ─────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1DB954 0%, #17a349 100%);
    color: #fff;
    border: none;
    border-radius: 50px;
    font-size: 1.05em;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 0.7em 2em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(29, 185, 84, 0.35);
    text-transform: uppercase;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 28px rgba(29, 185, 84, 0.55);
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0px);
}
.stButton > button[kind="primary"]:disabled {
    background: #2a2a3a;
    box-shadow: none;
    color: #555;
}

/* ── Secondary buttons ─────────────────────────────────────────────────── */
.stButton > button[kind="secondary"] {
    background: #1a1a2e;
    color: #ccc;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
}

/* ── Sliders ───────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {
    margin-top: 4px;
}
.stSlider [data-testid="stTickBar"] {
    display: none;
}
[data-testid="stSliderThumb"] {
    background: #1DB954 !important;
    box-shadow: 0 0 0 3px rgba(29,185,84,0.25);
}
[data-baseweb="slider"] [role="slider"] {
    background: #1DB954;
}
[data-baseweb="slider"] > div > div:nth-child(2) {
    background: #1DB954;
}

/* ── Selectbox / dropdowns ─────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: #13131f;
    border: 1px solid #252535;
    border-radius: 8px;
    color: #E0E0E0;
}
.stSelectbox > div > div:hover {
    border-color: #1DB954;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #13131f;
    color: #E0E0E0;
}

/* ── Radio buttons ─────────────────────────────────────────────────────── */
.stRadio > div {
    gap: 6px;
}
.stRadio > div > label {
    background: #13131f;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 6px 14px;
    color: #ccc;
    cursor: pointer;
    transition: all 0.2s;
}
.stRadio > div > label:has(input:checked) {
    border-color: #1DB954;
    color: #1DB954;
    background: rgba(29,185,84,0.08);
}

/* ── Checkboxes ────────────────────────────────────────────────────────── */
.stCheckbox > label {
    color: #C0C0C0;
}

/* ── Select-slider ─────────────────────────────────────────────────────── */
.stSelectSlider [data-baseweb="slider"] > div > div:nth-child(2) {
    background: #1DB954;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1em 0;
}

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 1.9em !important;
    font-weight: 800 !important;
    color: #1DB954 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78em !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="metric-container"] {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1em 1.2em;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1DB954, #17a349);
    border-radius: 8px;
    height: 10px !important;
}
.stProgress > div > div {
    background: #1a1a2e;
    border-radius: 8px;
    height: 10px !important;
}

/* ── Expander ──────────────────────────────────────────────────────────── */
.stExpander {
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    background: #0f0f1a;
}
.stExpander > details > summary {
    color: #888;
    font-size: 0.85em;
}

/* ── Dataframe / tables ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    background: #0f0f1a;
}

/* ── Info / warning / error / success boxes ────────────────────────────── */
.stAlert {
    border-radius: 10px;
}
[data-baseweb="notification"] {
    background: #0f0f1a !important;
}

/* ── Spinner ───────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #1DB954 !important;
}

/* ── Caption ───────────────────────────────────────────────────────────── */
.stCaptionContainer p {
    color: #666 !important;
    font-size: 0.8em;
}

/* ── Custom card containers ────────────────────────────────────────────── */
.card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1.4em 1.6em 1em;
    margin-bottom: 1em;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.section-label {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #1DB954;
    border-left: 3px solid #1DB954;
    padding-left: 0.6em;
    margin-bottom: 0.8em;
    margin-top: 0.2em;
}
.section-label-blue {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #0F62FE;
    border-left: 3px solid #0F62FE;
    padding-left: 0.6em;
    margin-bottom: 0.8em;
    margin-top: 0.2em;
}

/* ── Result banners ────────────────────────────────────────────────────── */
.result-popular {
    background: linear-gradient(135deg, rgba(29,185,84,0.12), rgba(23,163,73,0.06));
    border: 1px solid rgba(29,185,84,0.4);
    border-radius: 14px;
    padding: 1.4em 1.6em;
    margin-bottom: 1.2em;
    display: flex;
    align-items: center;
    gap: 1em;
    box-shadow: 0 0 30px rgba(29,185,84,0.12);
}
.result-not-popular {
    background: linear-gradient(135deg, rgba(255,152,0,0.1), rgba(255,120,0,0.05));
    border: 1px solid rgba(255,152,0,0.35);
    border-radius: 14px;
    padding: 1.4em 1.6em;
    margin-bottom: 1.2em;
    display: flex;
    align-items: center;
    gap: 1em;
}
.result-icon { font-size: 2.4em; }
.result-verdict {
    font-size: 1.4em;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #F0F0F0;
}
.result-sub {
    font-size: 0.85em;
    color: #888;
    margin-top: 2px;
}
.popular-verdict { color: #1DB954; }
.notpopular-verdict { color: #FF9800; }

/* ── KPI grid ──────────────────────────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin: 1em 0;
}
.kpi-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1em 0.8em;
    text-align: center;
}
.kpi-value {
    font-size: 1.8em;
    font-weight: 800;
    color: #1DB954;
    line-height: 1;
}
.kpi-value-blue { color: #0F62FE; }
.kpi-value-gray { color: #888; }
.kpi-label {
    font-size: 0.68em;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin-top: 4px;
}
.kpi-unit {
    font-size: 0.72em;
    color: #444;
    margin-top: 1px;
}

/* ── Score bar ─────────────────────────────────────────────────────────── */
.score-bar-label {
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin-bottom: 4px;
}
.score-bar-wrap {
    background: #1a1a2e;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1DB954, #17a349);
    transition: width 0.6s ease;
}

/* ── Confidence note ───────────────────────────────────────────────────── */
.conf-note {
    font-size: 0.8em;
    color: #666;
    background: #0d0d18;
    border: 1px solid #1a1a28;
    border-radius: 8px;
    padding: 0.7em 1em;
    margin-top: 0.8em;
    line-height: 1.5;
}
.conf-note strong { color: #aaa; }

/* ── Placeholder card (before predict) ─────────────────────────────────── */
.placeholder-card {
    background: #0d0d18;
    border: 1px dashed #252535;
    border-radius: 14px;
    padding: 2em 1.5em;
    text-align: center;
    color: #555;
    margin-top: 0.5em;
}
.placeholder-icon { font-size: 2.5em; margin-bottom: 0.3em; }
.placeholder-text { font-size: 0.9em; color: #555; }
.guide-table-wrap {
    margin-top: 1.2em;
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.82em;
}
.guide-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    color: #aaa;
}
.guide-table-wrap th {
    background: #111118;
    color: #666;
    text-transform: uppercase;
    font-size: 0.75em;
    letter-spacing: 0.8px;
    padding: 8px 12px;
    border-bottom: 1px solid #1e1e2e;
    text-align: left;
}
.guide-table-wrap td {
    padding: 7px 12px;
    border-bottom: 1px solid #131320;
}
.guide-table-wrap tr:nth-child(even) td { background: #0d0d18; }
.guide-table-wrap tr:last-child td { border-bottom: none; }
.guide-table-wrap td:first-child { color: #1DB954; font-weight: 600; }

/* ── EDA dashboard ─────────────────────────────────────────────────────── */
.eda-section-header {
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #0F62FE;
    border-left: 3px solid #0F62FE;
    padding-left: 0.7em;
    margin: 1.5em 0 0.8em;
}
.eda-intro {
    background: #0d0d18;
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    padding: 1em 1.3em;
    font-size: 0.88em;
    color: #888;
    margin-bottom: 1.2em;
    line-height: 1.6;
}
.eda-caption {
    font-size: 0.78em;
    color: #555;
    text-align: center;
    margin-top: 4px;
    padding: 0 0.5em 0.5em;
    font-style: italic;
}
.img-wrap {
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    margin-bottom: 0.3em;
}

/* ── About tab ─────────────────────────────────────────────────────────── */
.about-section {
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #1DB954;
    border-left: 3px solid #1DB954;
    padding-left: 0.7em;
    margin: 1.5em 0 0.8em;
}
.arch-block {
    background: #0d0d18;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.2em 1.5em;
    font-family: 'Courier New', monospace;
    font-size: 0.82em;
    color: #a0a0c0;
    line-height: 1.8;
    overflow-x: auto;
}
.arch-phase {
    color: #1DB954;
    font-weight: 700;
    font-size: 0.85em;
    letter-spacing: 0.5px;
}
.arch-arrow { color: #0F62FE; }
.about-table-wrap {
    border: 1px solid #1a1a2e;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.85em;
    margin: 0.5em 0 1em;
}
.about-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    color: #aaa;
}
.about-table-wrap th {
    background: #111118;
    color: #666;
    text-transform: uppercase;
    font-size: 0.72em;
    letter-spacing: 0.8px;
    padding: 9px 14px;
    border-bottom: 1px solid #1e1e2e;
    text-align: left;
}
.about-table-wrap td {
    padding: 8px 14px;
    border-bottom: 1px solid #131320;
}
.about-table-wrap tr:nth-child(even) td { background: #0d0d18; }
.about-table-wrap tr:last-child td { border-bottom: none; }
.about-table-wrap td:first-child { color: #E0E0E0; font-weight: 500; }
.about-table-wrap td code {
    background: #1a1a2e;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    color: #1DB954;
}
.student-box {
    background: linear-gradient(135deg, rgba(15,98,254,0.08), rgba(29,185,84,0.06));
    border: 1px solid #1e2e3e;
    border-radius: 12px;
    padding: 1.2em 1.5em;
    margin-top: 1.5em;
    font-size: 0.85em;
}
.student-box .row { display: flex; gap: 1em; margin: 4px 0; }
.student-box .lbl { color: #555; width: 80px; flex-shrink: 0; }
.student-box .val { color: #C0C0C0; }
.url-chip {
    display: inline-block;
    background: #0d0d18;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: monospace;
    font-size: 0.8em;
    color: #888;
    margin: 2px 0;
}

/* ── Header gradient title ─────────────────────────────────────────────── */
.app-title {
    font-size: 2.4em;
    font-weight: 900;
    background: linear-gradient(90deg, #1DB954 0%, #0F62FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 0.1em;
}
.app-tagline {
    color: #555;
    font-size: 0.95em;
    margin-bottom: 0.6em;
    letter-spacing: 0.3px;
}
.header-divider {
    height: 1px;
    background: linear-gradient(90deg, #1DB954 0%, #0F62FE 50%, transparent 100%);
    margin: 0.8em 0 1.4em;
    border: none;
}
.backend-error {
    background: rgba(255,60,60,0.08);
    border: 1px solid rgba(255,60,60,0.3);
    border-radius: 10px;
    padding: 0.9em 1.2em;
    font-size: 0.88em;
    color: #ff8080;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="app-title">🎵 Spotify Popularity Predictor</div>
<div class="app-tagline">Predict a song's commercial success from its audio DNA — powered by XGBoost</div>
<div class="header-divider"></div>
""", unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="backend-error">
    <strong>⚠️ Backend offline</strong> — cannot reach FastAPI at <code>{API_URL}</code><br>
    Start it with: <code>python -m uvicorn app.api:app --reload --port 8000</code> then refresh.
    </div>
    """, unsafe_allow_html=True)

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

    # ── Two-column layout: inputs on the left, results on the right ───────────
    col_inputs, col_results = st.columns([1.2, 1], gap="large")

    with col_inputs:

        # ── Genre & metadata ──────────────────────────────────────────────────
        st.markdown('<div class="section-label">Genre &amp; Metadata</div>', unsafe_allow_html=True)
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
                help="Track length. 3-4 min is typical for popular songs.",
            )
        duration_ms = int(duration_min * 60_000)

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # ── Audio features ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Audio Features</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # ── Musical structure ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Musical Structure</div>', unsafe_allow_html=True)
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
                help="Musical key. 0=C, 1=C#, ... 11=B.",
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

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # ── Predict button ────────────────────────────────────────────────────
        predict_btn = st.button(
            "Predict popularity", type="primary", use_container_width=True,
            disabled=not backend_ok,
        )
        st.caption("Model uses XGBoost trained on 106,907 Spotify tracks · random_state=42")

    # ── Right column: results ─────────────────────────────────────────────────
    with col_results:

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

            with st.spinner("Asking the model ..."):
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
                    st.markdown(f"""
                    <div class="result-popular">
                        <div class="result-icon">✅</div>
                        <div>
                            <div class="result-verdict popular-verdict">POPULAR</div>
                            <div class="result-sub">This song is predicted to be a commercial hit</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-not-popular">
                        <div class="result-icon">🎯</div>
                        <div>
                            <div class="result-verdict notpopular-verdict">NOT POPULAR</div>
                            <div class="result-sub">Below the popularity threshold of 70 / 100</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Key metrics — custom KPI cards ────────────────────────────
                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{score:.1f}</div>
                        <div class="kpi-label">Predicted Score</div>
                        <div class="kpi-unit">out of 100</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value kpi-value-blue">{confidence*100:.1f}%</div>
                        <div class="kpi-label">Confidence</div>
                        <div class="kpi-unit">classifier prob.</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value kpi-value-gray">70</div>
                        <div class="kpi-label">Threshold</div>
                        <div class="kpi-unit">popularity cutoff</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Popularity bar ────────────────────────────────────────────
                st.markdown(f"""
                <div class="score-bar-label">Score on the 0–100 scale</div>
                <div class="score-bar-wrap">
                    <div class="score-bar-fill" style="width:{min(score,100):.1f}%;"></div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence note ───────────────────────────────────────────
                st.markdown(f"""
                <div class="conf-note">
                    The classifier gives a <strong>{confidence*100:.1f}% probability</strong>
                    that this song is popular (popularity &ge; 70).
                    The regressor independently predicts a score of <strong>{score:.1f} / 100</strong>.
                </div>
                """, unsafe_allow_html=True)

                # ── Feature summary ───────────────────────────────────────────
                with st.expander("Show the features you submitted"):
                    import pandas as pd
                    feature_df = pd.DataFrame([payload]).T
                    feature_df.columns = ["Value"]
                    st.dataframe(feature_df, use_container_width=True)

        else:
            # Placeholder before the user clicks Predict
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon">🎵</div>
                <div class="placeholder-text">
                    Adjust the sliders on the left to describe your song,<br>
                    then click <strong style="color:#1DB954;">Predict popularity</strong>.
                </div>
            </div>
            <div class="guide-table-wrap">
                <table>
                    <thead>
                        <tr><th>Feature</th><th>Low value</th><th>High value</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Danceability</td><td>Hard to dance to</td><td>Easy to dance to</td></tr>
                        <tr><td>Energy</td><td>Calm, soft</td><td>Intense, loud</td></tr>
                        <tr><td>Valence</td><td>Sad / angry</td><td>Happy / euphoric</td></tr>
                        <tr><td>Acousticness</td><td>Electronic / synthetic</td><td>Acoustic instruments</td></tr>
                        <tr><td>Speechiness</td><td>Pure music</td><td>Mostly spoken words</td></tr>
                        <tr><td>Instrumentalness</td><td>Has vocals</td><td>No vocals</td></tr>
                        <tr><td>Liveness</td><td>Studio recording</td><td>Live audience</td></tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:

    st.markdown("""
    <div class="eda-intro">
        <strong style="color:#C0C0C0;">What is EDA?</strong> Exploratory Data Analysis is the step we do
        <em>before</em> any modelling — it means looking at the raw data to understand its shape,
        spot problems, and find patterns. All charts below were generated by <code>train.py</code>
        directly from the Spotify dataset. Nothing was changed before plotting — you see the data exactly as
        it came from Kaggle.
    </div>
    """, unsafe_allow_html=True)

    # Helper: show a plot if the file exists; otherwise show a placeholder
    def _show_plot(filename: str, caption: str) -> None:
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            st.image(path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eda-caption">{caption}</div>', unsafe_allow_html=True)
        else:
            st.info(
                f"**{filename}** not found.  \n"
                "Run `python train.py` to generate EDA plots."
            )

    # ── Section 1: Target variable ────────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Target Variable — Popularity</div>', unsafe_allow_html=True)
    _show_plot(
        "01_popularity_distribution.png",
        "Left: raw distribution of popularity scores (0-100). "
        "Right: class balance — only 5.1% of songs cross the threshold of 70.",
    )

    # ── Section 2: Feature distributions ─────────────────────────────────────
    st.markdown('<div class="eda-section-header">Audio Feature Distributions</div>', unsafe_allow_html=True)
    _show_plot(
        "02_feature_distributions.png",
        "Histogram of each numeric feature. "
        "Helps spot skewed or bimodal distributions that influence model choice.",
    )

    # ── Section 3: Correlations ───────────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Correlations</div>', unsafe_allow_html=True)
    _show_plot(
        "03_correlation_heatmap.png",
        "Pearson correlation matrix (lower triangle only). "
        "Red = positive correlation, blue = negative. Bottom row shows correlation with popularity.",
    )

    # ── Section 4: Outliers ────────────────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Outlier Detection</div>', unsafe_allow_html=True)
    _show_plot(
        "04_outlier_boxplots.png",
        "Boxplots for key features. Dots beyond the whiskers are outliers. "
        "We use RobustScaler in the Pipeline to handle these without removing rows.",
    )

    # ── Section 5: Feature–target relationships ───────────────────────────────
    st.markdown('<div class="eda-section-header">Feature-Target Relationships</div>', unsafe_allow_html=True)
    _show_plot(
        "05_feature_target_scatter.png",
        "Each audio feature plotted against the raw popularity score (5,000-row sample). "
        "Weak slopes confirm no single feature dominates — we need an ensemble model.",
    )

    # ── Section 6: Genre popularity ───────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Genre Analysis</div>', unsafe_allow_html=True)
    _show_plot(
        "06_genre_popularity.png",
        "Top 25 genres by mean popularity. "
        "Genre is the strongest single predictor — pop, dance, and hip-hop score highest.",
    )

    # ── Section 7: Model evaluation plots ─────────────────────────────────────
    st.markdown('<div class="eda-section-header">Model Evaluation — XGBoost (Best Model)</div>', unsafe_allow_html=True)
    c_left, c_right = st.columns(2)
    with c_left:
        _show_plot(
            "07_confusion_matrix_xgb.png",
            "Confusion matrix on the held-out test set. "
            "True positives = popular songs correctly identified.",
        )
        _show_plot(
            "08_roc_curve_xgb.png",
            "ROC curve: AUC = 0.920. Random classifier would score 0.50.",
        )
    with c_right:
        _show_plot(
            "09_predicted_vs_actual_xgb.png",
            "Predicted vs actual popularity. "
            "Perfect predictions would lie on the red diagonal.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:

    st.markdown('<div class="about-section">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-block">
        <span class="arch-phase">OFFLINE</span> &mdash; runs once on your machine<br>
        <span style="color:#444;">&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</span><br>
        dataset.csv <span class="arch-arrow">&rarr;</span> train.py <span class="arch-arrow">&rarr;</span> EDA plots + classifier.pkl + regressor.pkl<br><br>
        <span class="arch-phase">RUNTIME</span> &mdash; this app, always on<br>
        <span style="color:#444;">&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</span><br>
        You <span class="arch-arrow">&rarr;</span> Streamlit (this page) <span class="arch-arrow">&rarr;</span> HTTP POST <span class="arch-arrow">&rarr;</span> FastAPI <span class="arch-arrow">&rarr;</span> .pkl <span class="arch-arrow">&rarr;</span> prediction
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section">Two ML Tasks</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-table-wrap">
        <table>
            <thead><tr><th>Task</th><th>Target</th><th>Best model</th><th>Score</th></tr></thead>
            <tbody>
                <tr><td>Classification</td><td>popularity &ge; 70 &rarr; popular (1 / 0)</td><td>XGBoost (tuned)</td><td>F1 = 0.437 &nbsp;|&nbsp; AUC = 0.920</td></tr>
                <tr><td>Regression</td><td>exact popularity score 0&ndash;100</td><td>XGBoost (tuned)</td><td>R&sup2; = 0.380 &nbsp;|&nbsp; RMSE = 16.7</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section">Model Ladder</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-table-wrap">
        <table>
            <thead><tr><th>#</th><th>Model</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td>1</td><td>DummyClassifier / DummyRegressor</td><td>Baseline floor — must beat this</td></tr>
                <tr><td>2</td><td>Logistic Regression / Ridge</td><td>Linear relationship benchmark</td></tr>
                <tr><td>3</td><td>Decision Tree (depth=6)</td><td>Non-linear, interpretable, overfits without depth control</td></tr>
                <tr><td>4</td><td>XGBoost + RandomizedSearchCV</td><td>Gradient boosting — best performer</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section">Key Design Decisions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-table-wrap">
        <table>
            <thead><tr><th>Decision</th><th>Why</th></tr></thead>
            <tbody>
                <tr><td><code>random_state=42</code> everywhere</td><td>Fully reproducible results across runs</td></tr>
                <tr><td>scikit-learn <code>Pipeline</code></td><td>No data leakage — scaler fitted on training data only</td></tr>
                <tr><td><code>RobustScaler</code> not <code>StandardScaler</code></td><td>Handles outliers in loudness / tempo without removing rows</td></tr>
                <tr><td>Models saved as <code>.pkl</code></td><td>Backend loads once at startup — never retrains live</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="about-section">Links</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="student-box">
        <div class="row"><span class="lbl">Student</span><span class="val">Herend Amila</span></div>
        <div class="row"><span class="lbl">Course</span><span class="val">Modelling in Advanced Data Analytics &mdash; FELU</span></div>
        <div class="row"><span class="lbl">Dataset</span>
            <span class="val"><a href="https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
                style="color:#1DB954;">Spotify Tracks Dataset</a>
            &nbsp;&mdash;&nbsp; Maharshi Pandya, Kaggle 2022 &nbsp;&mdash;&nbsp;
            ~114,000 tracks, 21 audio features, 114 genres</span></div>
        <div class="row"><span class="lbl">Backend</span>
            <span class="val"><span class="url-chip">{API_URL}</span></span></div>
        <div class="row"><span class="lbl">API docs</span>
            <span class="val"><a href="{API_URL}/docs" style="color:#0F62FE;">{API_URL}/docs</a></span></div>
    </div>
    """, unsafe_allow_html=True)
