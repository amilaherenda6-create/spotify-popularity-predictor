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
import pandas as pd
import numpy as np

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

@st.cache_data
def _load_dataset() -> pd.DataFrame:
    url = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
    return pd.read_csv(url)

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
# THEME STATE
# =============================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# =============================================================================
# THEME CSS
# All colors live in two dicts (_DARK / _LIGHT). _build_css() renders them
# into a single <style> block — no duplicated CSS rules, only the values swap.
# =============================================================================

_DARK = dict(
    bg_app="#0A0A0F",     bg_card="#111118",    bg_card2="#0d0d18",
    border="#1e1e2e",     text="#E0E0E0",        text_sec="#C0C0C0",
    text_muted="#888",    text_dim="#666",        text_vdim="#555",
    text_edim="#444",     tab_list="#111118",    tab_text="#888",
    tab_hover="#1a1a28",  input_bg="#13131f",    input_border="#252535",
    progress_bg="#1a1a2e", expander_bg="#0f0f1a", sidebar_bg="#0d0d18",
    sidebar_border="#1a1a2e", result_text="#F0F0F0", result_sub="#888",
    table_hdr="#111118",  table_alt="#0d0d18",   table_row_bdr="#131320",
    table_text="#aaa",    arch_bg="#0d0d18",     arch_text="#a0a0c0",
    arch_sep="#444",      code_bg="#1a1a2e",     student_border="#1e2e3e",
    chip_bg="#0d0d18",    error_bg="rgba(255,60,60,0.08)",
    error_border="rgba(255,60,60,0.30)", error_text="#ff8080",
    toggle_bg="#1a1a28",  toggle_border="#252535", toggle_text="#aaa",
    toggle_hover="#222235", img_shadow="rgba(0,0,0,0.40)",
    conf_bg="#0d0d18",    conf_border="#1a1a28", conf_text="#666",
    conf_strong="#aaa",   metric_bg="#111118",   kpi_bg="#111118",
    score_bg="#1a1a2e",   eda_intro_bg="#0d0d18", eda_img_bdr="#1a1a2e",
    h_color="#F0F0F0",
    pop_bg="linear-gradient(135deg,rgba(29,185,84,0.12),rgba(23,163,73,0.06))",
    pop_bdr="rgba(29,185,84,0.40)", pop_glow="0 0 30px rgba(29,185,84,0.12)",
    nop_bg="linear-gradient(135deg,rgba(255,152,0,0.10),rgba(255,120,0,0.05))",
    nop_bdr="rgba(255,152,0,0.35)",
    student_bg="linear-gradient(135deg,rgba(15,98,254,0.08),rgba(29,185,84,0.06))",
)

_LIGHT = dict(
    bg_app="#F5F5F7",     bg_card="#FFFFFF",     bg_card2="#F8F8FA",
    border="#E0E0E0",     text="#1A1A2E",         text_sec="#444",
    text_muted="#777",    text_dim="#999",         text_vdim="#AAA",
    text_edim="#BBB",     tab_list="#EAEAEA",     tab_text="#777",
    tab_hover="#D8D8D8",  input_bg="#FFFFFF",     input_border="#CCC",
    progress_bg="#E0E0E0", expander_bg="#F5F5F7", sidebar_bg="#FFFFFF",
    sidebar_border="#E0E0E0", result_text="#1A1A2E", result_sub="#777",
    table_hdr="#EEEEEE",  table_alt="#F5F5F7",   table_row_bdr="#F0F0F0",
    table_text="#666",    arch_bg="#F0F0F5",      arch_text="#555577",
    arch_sep="#CCC",      code_bg="#E8E8F0",      student_border="#C8D8E8",
    chip_bg="#F0F0F5",    error_bg="rgba(255,60,60,0.04)",
    error_border="rgba(255,60,60,0.15)", error_text="#cc4444",
    toggle_bg="#FFFFFF",  toggle_border="#E0E0E0", toggle_text="#555",
    toggle_hover="#F0F0F5", img_shadow="rgba(0,0,0,0.08)",
    conf_bg="#F8F8FA",    conf_border="#E8E8EA", conf_text="#777",
    conf_strong="#555",   metric_bg="#FFFFFF",   kpi_bg="#FFFFFF",
    score_bg="#E0E0E0",   eda_intro_bg="#F0F0F5", eda_img_bdr="#E0E0E0",
    h_color="#1A1A2E",
    pop_bg="linear-gradient(135deg,rgba(29,185,84,0.07),rgba(23,163,73,0.03))",
    pop_bdr="rgba(29,185,84,0.30)", pop_glow="0 2px 16px rgba(29,185,84,0.06)",
    nop_bg="linear-gradient(135deg,rgba(255,152,0,0.07),rgba(255,120,0,0.03))",
    nop_bdr="rgba(255,152,0,0.25)",
    student_bg="linear-gradient(135deg,rgba(15,98,254,0.04),rgba(29,185,84,0.03))",
)


def _build_css(t: dict) -> str:
    return f"""
<style>
/* ── Base ─────────────────────────────────────────────────────────────── */
.stApp {{ background-color: {t['bg_app']}; color: {t['text']}; }}
.stApp > header {{ background-color: transparent; }}
[data-testid="stAppViewContainer"] {{ background-color: {t['bg_app']}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}

/* ── Typography ────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{ color: {t['h_color']} !important; }}
p, li, label {{ color: {t['text_sec']}; }}
.stMarkdown p {{ color: {t['text_sec']}; }}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {t['sidebar_bg']};
    border-right: 1px solid {t['sidebar_border']};
}}

/* ── Tabs — pill nav ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-panel"] {{ background: transparent; padding: 0; }}
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px; background: {t['tab_list']}; border-radius: 14px;
    padding: 5px 6px; border: 1px solid {t['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px; color: {t['tab_text']}; padding: 8px 22px;
    font-weight: 500; font-size: 0.9em; background: transparent;
    border: none; transition: all 0.2s ease; letter-spacing: 0.3px;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {t['text']}; background: {t['tab_hover']}; }}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg,#1DB954,#17a349) !important;
    color: #fff !important; font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(29,185,84,0.35);
}}

/* ── Primary button ──────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {{
    background: linear-gradient(135deg,#1DB954 0%,#17a349 100%) !important;
    color: #fff !important; border: none !important; border-radius: 50px !important;
    font-size: 1.05em !important; font-weight: 700 !important; letter-spacing: 0.8px !important;
    padding: 0.7em 2em !important; transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(29,185,84,0.35) !important; text-transform: uppercase !important;
}}
[data-testid="baseButton-primary"]:hover {{
    transform: translateY(-2px) !important; box-shadow: 0 6px 28px rgba(29,185,84,0.55) !important;
}}
[data-testid="baseButton-primary"]:active {{ transform: translateY(0) !important; }}
[data-testid="baseButton-primary"]:disabled {{
    background: {t['tab_hover']} !important; box-shadow: none !important; color: {t['text_vdim']} !important;
}}

/* ── Theme toggle ────────────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"] {{
    background: {t['toggle_bg']} !important;
    color: {t['text']} !important;
    border: 1.5px solid #1DB954 !important;
    border-radius: 50px !important;
    font-size: 0.85em !important;
    font-weight: 700 !important;
    padding: 7px 20px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(29,185,84,0.20) !important;
    width: 100% !important;
    white-space: nowrap !important;
}}
[data-testid="baseButton-secondary"]:hover {{
    background: rgba(29,185,84,0.12) !important;
    border-color: #1DB954 !important;
    color: #1DB954 !important;
    box-shadow: 0 4px 20px rgba(29,185,84,0.35) !important;
    transform: translateY(-2px) !important;
}}

/* ── rs ─────────────────────────────────────────────────────────────── */
[data-testid="stTickBar"] {{ display: none; }}
[data-testid="stSliderThumb"] {{
    background: #1DB954 !important;
    box-shadow: 0 0 0 3px rgba(29,185,84,0.25);
}}
[data-baseweb="slider"] [role="slider"] {{ background: #1DB954; }}
[data-baseweb="slider"] > div > div:nth-child(2) {{ background: #1DB954; }}

/* ── Selectbox ───────────────────────────────────────────────────────────── */
.stSelectbox > div > div {{
    background: {t['input_bg']}; border: 1px solid {t['input_border']};
    border-radius: 8px; color: {t['text']};
}}
.stSelectbox > div > div:hover {{ border-color: #1DB954; }}
.stSelectbox [data-baseweb="select"] > div {{ background: {t['input_bg']}; color: {t['text']}; }}

/* ── Radio ───────────────────────────────────────────────────────────────── */
.stRadio > div {{ gap: 6px; }}
.stRadio > div > label {{
    background: {t['input_bg']}; border: 1px solid {t['input_border']};
    border-radius: 8px; padding: 6px 14px; color: {t['text_sec']};
    cursor: pointer; transition: all 0.2s;
}}
.stRadio > div > label:has(input:checked) {{
    border-color: #1DB954; color: #1DB954; background: rgba(29,185,84,0.08);
}}

/* ── Checkbox ────────────────────────────────────────────────────────────── */
.stCheckbox > label {{ color: {t['text_sec']}; }}

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div > div {{
    background: linear-gradient(90deg,#1DB954,#17a349);
    border-radius: 8px; height: 10px !important;
}}
.stProgress > div > div {{
    background: {t['progress_bg']}; border-radius: 8px; height: 10px !important;
}}

/* ── Expander ────────────────────────────────────────────────────────────── */
.stExpander {{
    border: 1px solid {t['border']}; border-radius: 10px;
    background: {t['expander_bg']};
}}
.stExpander > details > summary {{ color: {t['text_muted']}; font-size: 0.85em; }}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] > div {{
    border: 1px solid rgba(29, 185, 84, 0.25) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] th {{
    background-color: rgba(29, 185, 84, 0.12) !important;
    color: #1DB954 !important;
    border-bottom: 1px solid rgba(29, 185, 84, 0.25) !important;
    font-weight: 700 !important;
}}
[data-testid="stDataFrame"] td {{
    color: #C0C0C0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}}
[data-testid="stDataFrame"] tr:last-child td {{
    color: #1DB954 !important;
    font-weight: 600 !important;
    background: rgba(29, 185, 84, 0.06) !important;
}}
[data-testid="stDataFrame"] tr:hover td {{
    background: rgba(29, 185, 84, 0.04) !important;
}}

/* ── Divider / caption / spinner ────────────────────────────────────────── */
hr {{ border: none; border-top: 1px solid {t['border']}; margin: 1em 0; }}
.stCaptionContainer p {{ color: {t['text_dim']} !important; font-size: 0.8em; }}
.stSpinner > div {{ border-top-color: #1DB954 !important; }}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    font-size: 1.9em !important; font-weight: 800 !important; color: #1DB954 !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78em !important; color: {t['text_muted']} !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}}
[data-testid="metric-container"] {{
    background: {t['metric_bg']}; border: 1px solid {t['border']};
    border-radius: 12px; padding: 1em 1.2em;
}}

/* ─── CUSTOM HTML ELEMENTS ──────────────────────────────────────────────── */

.app-title {{
    font-size: 2.4em; font-weight: 900;
    background: linear-gradient(90deg,#1DB954 0%,#0F62FE 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 0.1em;
}}
.app-tagline {{ color: {t['text_dim']}; font-size: 0.95em; margin-bottom: 0.6em; letter-spacing: 0.3px; }}
.header-divider {{
    height: 1px;
    background: linear-gradient(90deg,#1DB954 0%,#0F62FE 50%,transparent 100%);
    margin: 0.8em 0 1.4em; border: none;
}}
.backend-error {{
    background: {t['error_bg']}; border: 1px solid {t['error_border']};
    border-radius: 10px; padding: 0.9em 1.2em;
    font-size: 0.88em; color: {t['error_text']}; margin-bottom: 1em;
}}
.section-label {{
    font-size: 0.72em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #1DB954; border-left: 3px solid #1DB954;
    padding-left: 0.6em; margin-bottom: 0.8em; margin-top: 0.2em;
}}
.result-popular {{
    background: {t['pop_bg']}; border: 1px solid {t['pop_bdr']};
    border-radius: 14px; padding: 1.4em 1.6em; margin-bottom: 1.2em;
    display: flex; align-items: center; gap: 1em; box-shadow: {t['pop_glow']};
}}
.result-not-popular {{
    background: {t['nop_bg']}; border: 1px solid {t['nop_bdr']};
    border-radius: 14px; padding: 1.4em 1.6em; margin-bottom: 1.2em;
    display: flex; align-items: center; gap: 1em;
}}
.result-icon {{ font-size: 2.4em; }}
.result-verdict {{ font-size: 1.4em; font-weight: 800; letter-spacing: 0.5px; color: {t['result_text']}; }}
.result-sub {{ font-size: 0.85em; color: {t['result_sub']}; margin-top: 2px; }}
.popular-verdict {{ color: #1DB954; }}
.notpopular-verdict {{ color: #FF9800; }}
.kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 1em 0; }}
.kpi-card {{
    background: {t['kpi_bg']}; border: 1px solid {t['border']};
    border-radius: 12px; padding: 1em 0.8em; text-align: center;
}}
.kpi-value      {{ font-size: 1.8em; font-weight: 800; color: #1DB954; line-height: 1; }}
.kpi-value-blue {{ color: #0F62FE; }}
.kpi-value-gray {{ color: {t['text_muted']}; }}
.kpi-label      {{ font-size: 0.68em; text-transform: uppercase; letter-spacing: 1px; color: {t['text_dim']}; margin-top: 4px; }}
.kpi-unit       {{ font-size: 0.72em; color: {t['text_edim']}; margin-top: 1px; }}
.score-bar-label {{ font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; color: {t['text_dim']}; margin-bottom: 4px; }}
.score-bar-wrap {{ background: {t['score_bg']}; border-radius: 8px; height: 12px; overflow: hidden; }}
.score-bar-fill {{
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg,#1DB954,#17a349); transition: width 0.6s ease;
}}
.conf-note {{
    font-size: 0.8em; color: {t['conf_text']}; background: {t['conf_bg']};
    border: 1px solid {t['conf_border']}; border-radius: 8px;
    padding: 0.7em 1em; margin-top: 0.8em; line-height: 1.5;
}}
.conf-note strong {{ color: {t['conf_strong']}; }}
.placeholder-card {{
    background: {t['bg_card2']}; border: 1px dashed {t['input_border']};
    border-radius: 14px; padding: 2em 1.5em;
    text-align: center; color: {t['text_vdim']}; margin-top: 0.5em;
}}
.placeholder-icon {{ font-size: 2.5em; margin-bottom: 0.3em; }}
.placeholder-text {{ font-size: 0.9em; color: {t['text_vdim']}; }}
.guide-table-wrap {{
    margin-top: 1.2em; border: 1px solid {t['border']};
    border-radius: 10px; overflow: hidden; font-size: 0.82em;
}}
.guide-table-wrap table {{ width: 100%; border-collapse: collapse; color: {t['table_text']}; }}
.guide-table-wrap th {{
    background: {t['table_hdr']}; color: {t['text_dim']};
    text-transform: uppercase; font-size: 0.75em; letter-spacing: 0.8px;
    padding: 8px 12px; border-bottom: 1px solid {t['border']}; text-align: left;
}}
.guide-table-wrap td {{ padding: 7px 12px; border-bottom: 1px solid {t['table_row_bdr']}; }}
.guide-table-wrap tr:nth-child(even) td {{ background: {t['table_alt']}; }}
.guide-table-wrap tr:last-child td {{ border-bottom: none; }}
.guide-table-wrap td:first-child {{ color: #1DB954; font-weight: 600; }}
.eda-section-header {{
    font-size: 0.78em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #0F62FE; border-left: 3px solid #0F62FE;
    padding-left: 0.7em; margin: 1.5em 0 0.8em;
}}
.eda-intro {{
    background: {t['eda_intro_bg']}; border: 1px solid {t['border']};
    border-radius: 10px; padding: 1em 1.3em; font-size: 0.88em;
    color: {t['text_muted']}; margin-bottom: 1.2em; line-height: 1.6;
}}
.eda-caption {{
    font-size: 0.78em; color: {t['text_vdim']}; text-align: center;
    margin-top: 4px; padding: 0 0.5em 0.5em; font-style: italic;
}}
.img-wrap {{
    border: 1px solid {t['eda_img_bdr']}; border-radius: 10px; overflow: hidden;
    box-shadow: 0 6px 24px {t['img_shadow']}; margin-bottom: 0.3em;
}}
.about-section {{
    font-size: 0.78em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #1DB954; border-left: 3px solid #1DB954;
    padding-left: 0.7em; margin: 1.5em 0 0.8em;
}}
.arch-block {{
    background: {t['arch_bg']}; border: 1px solid {t['border']}; border-radius: 10px;
    padding: 1.2em 1.5em; font-family: 'Courier New', monospace;
    font-size: 0.82em; color: {t['arch_text']}; line-height: 1.8; overflow-x: auto;
}}
.arch-phase {{ color: #1DB954; font-weight: 700; font-size: 0.85em; letter-spacing: 0.5px; }}
.arch-arrow {{ color: #0F62FE; }}
.arch-sep   {{ color: {t['arch_sep']}; }}
.about-table-wrap {{
    border: 1px solid {t['border']}; border-radius: 10px; overflow: hidden;
    font-size: 0.85em; margin: 0.5em 0 1em;
}}
.about-table-wrap table {{ width: 100%; border-collapse: collapse; color: {t['table_text']}; }}
.about-table-wrap th {{
    background: {t['table_hdr']}; color: {t['text_dim']}; text-transform: uppercase;
    font-size: 0.72em; letter-spacing: 0.8px; padding: 9px 14px;
    border-bottom: 1px solid {t['border']}; text-align: left;
}}
.about-table-wrap td {{ padding: 8px 14px; border-bottom: 1px solid {t['table_row_bdr']}; }}
.about-table-wrap tr:nth-child(even) td {{ background: {t['table_alt']}; }}
.about-table-wrap tr:last-child td {{ border-bottom: none; }}
.about-table-wrap td:first-child {{ color: {t['text']}; font-weight: 500; }}
.about-table-wrap td code {{
    background: {t['code_bg']}; padding: 1px 6px;
    border-radius: 4px; font-size: 0.9em; color: #1DB954;
}}
.student-box {{
    background: {t['student_bg']}; border: 1px solid {t['student_border']};
    border-radius: 12px; padding: 1.2em 1.5em; margin-top: 1.5em; font-size: 0.85em;
}}
.student-box .row {{ display: flex; gap: 1em; margin: 4px 0; }}
.student-box .lbl {{ color: {t['text_vdim']}; width: 80px; flex-shrink: 0; }}
.student-box .val {{ color: {t['text_sec']}; }}
.url-chip {{
    display: inline-block; background: {t['chip_bg']}; border: 1px solid {t['border']};
    border-radius: 6px; padding: 3px 10px; font-family: monospace;
    font-size: 0.8em; color: {t['text_muted']}; margin: 2px 0;
}}

/* ── Song Insights tab ──────────────────────────────────────────────────── */
.song-card {{
    background: {t['bg_card']}; border: 1px solid {t['border']};
    border-radius: 12px; padding: 0.9em 1.2em; margin-bottom: 0.55em;
    display: flex; align-items: center; justify-content: space-between; gap: 1em;
}}
.song-card-info {{ flex: 1; min-width: 0; }}
.song-title {{ font-weight: 700; color: {t['text']}; font-size: 0.93em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.song-meta {{ color: {t['text_muted']}; font-size: 0.78em; margin-top: 2px; }}
.pop-score-green {{ color: #1DB954; font-weight: 700; font-size: 0.85em; }}
.pop-score-orange {{ color: #FF9800; font-weight: 700; font-size: 0.85em; }}
.pop-score-gray {{ color: {t['text_muted']}; font-weight: 700; font-size: 0.85em; }}
.match-badge-green {{
    background: rgba(29,185,84,0.15); color: #1DB954;
    border: 1px solid rgba(29,185,84,0.30); border-radius: 20px;
    padding: 2px 10px; font-size: 0.76em; font-weight: 700; white-space: nowrap;
}}
.match-badge-blue {{
    background: rgba(15,98,254,0.15); color: #0F62FE;
    border: 1px solid rgba(15,98,254,0.30); border-radius: 20px;
    padding: 2px 10px; font-size: 0.76em; font-weight: 700; white-space: nowrap;
}}
.match-badge-gray {{
    background: {t['bg_card2']}; color: {t['text_muted']};
    border: 1px solid {t['border']}; border-radius: 20px;
    padding: 2px 10px; font-size: 0.76em; font-weight: 700; white-space: nowrap;
}}
.insight-box {{
    background: {t['bg_card2']}; border: 1px solid {t['border']};
    border-radius: 10px; padding: 0.85em 1.1em; font-size: 0.87em;
    color: {t['text_muted']}; margin: 0.8em 0; line-height: 1.55;
}}
.suggest-card {{
    background: {t['bg_card']}; border: 1px solid {t['border']};
    border-radius: 12px; padding: 1em 1.2em; margin-bottom: 0.8em;
}}
.suggest-header {{ font-weight: 700; color: {t['text']}; font-size: 0.93em; margin-bottom: 3px; }}
.suggest-values {{ font-size: 0.82em; color: {t['text_muted']}; margin-bottom: 6px; }}
.suggest-gain {{ font-size: 0.8em; color: #1DB954; font-weight: 700; }}
.total-score-box {{
    background: linear-gradient(135deg,rgba(29,185,84,0.12),rgba(15,98,254,0.08));
    border: 1px solid rgba(29,185,84,0.30); border-radius: 14px;
    padding: 1.1em 1.5em; text-align: center; margin-top: 1em;
}}
.total-score-val {{ font-size: 2.2em; font-weight: 900; color: #1DB954; }}
.total-score-lbl {{ font-size: 0.82em; color: {t['text_muted']}; margin-top: 2px; }}
</style>
"""


# Inject theme CSS
st.markdown(_build_css(_DARK if st.session_state.dark_mode else _LIGHT), unsafe_allow_html=True)

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stExpander {
        border: 1px solid rgba(29, 185, 84, 0.20) !important;
        background: rgba(29, 185, 84, 0.05) !important;
        border-radius: 10px;
    }
    .stExpander > details > summary {
        color: #1DB954 !important;
        font-size: 0.88em;
    }
    </style>
    """, unsafe_allow_html=True)

# Animations + compact layout
st.markdown("""
<style>
/* Fade in animation for the whole page */
.main {
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Predict button hover lift — primary only, never touches the toggle */
[data-testid="baseButton-primary"] {
    transition: all 0.3s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(29, 185, 84, 0.4) !important;
}

/* Result panel slide-in */
.stAlert {
    animation: slideIn 0.4s ease-out !important;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* Metric card hover */
[data-testid="metric-container"] {
    animation: popIn 0.3s ease-out !important;
    transition: transform 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
}

/* Slider transition */
.stSlider { transition: all 0.2s ease !important; }

/* Section header fade */
.stMarkdown h3 { animation: fadeIn 0.4s ease-in !important; }

/* Tab hover lift */
.stTabs [data-baseweb="tab"] { transition: all 0.2s ease !important; }
.stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px) !important; }

/* Main title animation */
.app-title {
    animation: titleDrop 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
@keyframes titleDrop {
    from { opacity: 0; transform: translateY(-24px); letter-spacing: 4px; }
    to   { opacity: 1; transform: translateY(0);     letter-spacing: -0.5px; }
}
.app-tagline {
    animation: fadeIn 1.1s ease-in !important;
}

/* ── Compact layout — fit predictor on one screen ─────────────────────── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
}
.stSlider {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    margin-bottom: -8px !important;
}
.stSelectbox, .stCheckbox { margin-bottom: -10px !important; }
.stMarkdown { margin-bottom: -5px !important; }
hr { margin: 0.3rem 0 !important; }
[data-testid="column"] { padding: 0 5px !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER — title left, theme toggle right
# =============================================================================
hdr_col, btn_col = st.columns([6, 1.4])

with hdr_col:
    st.markdown("""
    <div class="app-title">🎵 Spotify Popularity Predictor</div>
    <div class="app-tagline">Predict a song's commercial success from its audio DNA — powered by XGBoost</div>
    <div class="header-divider"></div>
    """, unsafe_allow_html=True)

with btn_col:
    st.markdown("<div style='padding-top:0.6em'></div>", unsafe_allow_html=True)
    toggle_label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
    if st.button(toggle_label, key="theme_toggle", help="Switch between dark and light theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

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
        resp = requests.get(f"{API_URL}/health", timeout=10)
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout):
        return False

backend_ok = _check_backend()
if not backend_ok:
    st.markdown(f"""
    <div class="backend-error">
    <strong>⏳ Backend is starting up</strong> — The free Render instance goes to sleep after inactivity.<br>
    Please wait <strong>30–60 seconds</strong> and refresh the page. It will wake up automatically!<br>
    <small style="color:#888">Backend URL: <code>{API_URL}</code></small>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab_predict, tab_insights, tab_explain, tab_eda, tab_about = st.tabs(
    ["🎵 Predictor", "🔍 Song Insights", "🧠 How It Works", "📊 EDA Dashboard", "ℹ️ About"]
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

        # ── Song selector — auto-fills sliders from a real Spotify song ───────
        _df_sel = _load_dataset()
        _genre_songs = _df_sel[_df_sel['track_genre'] == track_genre].copy()
        _genre_songs = _genre_songs.drop_duplicates(subset=['track_name', 'artists'])
        _genre_songs = _genre_songs.sort_values('popularity', ascending=False).head(50)
        _song_labels = ["— Select a song to auto-fill sliders (optional) —"] + [
            f"{row['track_name']} — {row['artists']} (popularity: {int(row['popularity'])})"
            for _, row in _genre_songs.iterrows()
        ]
        selected_song_label = st.selectbox(
            "🎵 Pick a real song from this genre (auto-fills sliders)",
            options=_song_labels,
            key=f"song_selector_{track_genre}",
        )
        if selected_song_label != "— Select a song to auto-fill sliders (optional) —":
            _sel_idx  = _song_labels.index(selected_song_label) - 1
            _sel_song = _genre_songs.iloc[_sel_idx]
            st.session_state['auto_danceability']      = float(_sel_song['danceability'])
            st.session_state['auto_energy']            = float(_sel_song['energy'])
            st.session_state['auto_valence']           = float(_sel_song['valence'])
            st.session_state['auto_acousticness']      = float(_sel_song['acousticness'])
            st.session_state['auto_speechiness']       = float(_sel_song['speechiness'])
            st.session_state['auto_instrumentalness']  = float(_sel_song['instrumentalness'])
            st.session_state['auto_liveness']          = float(_sel_song['liveness'])
            st.session_state['auto_loudness']          = float(_sel_song['loudness'])
            st.session_state['auto_tempo']             = float(_sel_song['tempo'])
            st.session_state['auto_key']               = int(_sel_song['key'])
            st.session_state['auto_mode']              = int(_sel_song['mode'])
            st.session_state['auto_time_signature']    = int(_sel_song['time_signature'])
            st.session_state['auto_duration']          = float(_sel_song['duration_ms']) / 60000
            st.session_state['auto_explicit']          = bool(_sel_song['explicit'])
            st.session_state['auto_filled']            = True
            col_msg, col_yt = st.columns([3, 1])
            with col_msg:
                st.success(f"✅ Sliders filled with values from **{_sel_song['track_name']}** by {_sel_song['artists']}")
            with col_yt:
                _yt_query = f"{_sel_song['track_name']} {_sel_song['artists']}".replace(" ", "+")
                st.markdown(
                    f"""<a href="https://www.youtube.com/results?search_query={_yt_query}"
                    target="_blank" style="display: inline-block; background: #FF0000;
                    color: white; padding: 0.5rem 1rem; border-radius: 8px;
                    text-decoration: none; font-weight: bold; margin-top: 0.3rem;">
                    ▶️ Listen on YouTube</a>""",
                    unsafe_allow_html=True,
                )

        c1, c2 = st.columns(2)
        with c1:
            explicit = st.checkbox(
                "Explicit content",
                value=st.session_state.get('auto_explicit', False),
                key="explicit_cb",
            )
        with c2:
            duration_min = st.slider(
                "Duration (minutes)", min_value=0.5, max_value=15.0,
                value=float(round(st.session_state.get('auto_duration', 3.5), 1)),
                step=0.1,
                help="Track length. 3-4 min is typical for popular songs.",
                key="duration_slider",
            )
        duration_ms = int(duration_min * 60_000)

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # ── Audio features ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Audio Features</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            danceability = st.slider(
                "Danceability", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_danceability', 0.65), 2)),
                step=0.01,
                help="How suitable for dancing. 0 = not danceable, 1 = very danceable.",
                key="danceability_slider",
            )
            energy = st.slider(
                "Energy", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_energy', 0.60), 2)),
                step=0.01,
                help="Intensity and activity. High energy = fast, loud, noisy.",
                key="energy_slider",
            )
            valence = st.slider(
                "Valence", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_valence', 0.50), 2)),
                step=0.01,
                help="Musical positiveness. 1 = happy/euphoric, 0 = sad/angry.",
                key="valence_slider",
            )
            acousticness = st.slider(
                "Acousticness", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_acousticness', 0.20), 2)),
                step=0.01,
                help="Confidence that the track is acoustic. 1 = fully acoustic.",
                key="acousticness_slider",
            )
        with c4:
            speechiness = st.slider(
                "Speechiness", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_speechiness', 0.05), 2)),
                step=0.01,
                help="Presence of spoken words. >0.66 = podcast/audiobook.",
                key="speechiness_slider",
            )
            instrumentalness = st.slider(
                "Instrumentalness", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_instrumentalness', 0.00), 2)),
                step=0.01,
                help="Likelihood of no vocals. >0.5 = probably instrumental.",
                key="instrumentalness_slider",
            )
            liveness = st.slider(
                "Liveness", min_value=0.0, max_value=1.0,
                value=float(round(st.session_state.get('auto_liveness', 0.10), 2)),
                step=0.01,
                help="Presence of an audience. >0.8 = probably a live recording.",
                key="liveness_slider",
            )
            loudness = st.slider(
                "Loudness (dB)", min_value=-60.0, max_value=5.0,
                value=float(round(st.session_state.get('auto_loudness', -7.0), 1)),
                step=0.5,
                help="Overall loudness. Typical studio tracks: -10 to -5 dB.",
                key="loudness_slider",
            )

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # ── Musical structure ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Musical Structure</div>', unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5:
            tempo = st.slider(
                "Tempo (BPM)", min_value=30.0, max_value=250.0,
                value=float(round(st.session_state.get('auto_tempo', 120.0), 1)),
                step=0.5,
                help="Beats per minute. 120 BPM = standard dance track.",
                key="tempo_slider",
            )
        with c6:
            _auto_key = st.session_state.get('auto_key', 5)
            key = st.selectbox(
                "Key",
                options=list(range(12)),
                format_func=lambda k: ["C","C#","D","D#","E","F",
                                       "F#","G","G#","A","A#","B"][k],
                index=int(_auto_key),
                help="Musical key. 0=C, 1=C#, ... 11=B.",
                key="key_selectbox",
            )
        with c7:
            _auto_mode = st.session_state.get('auto_mode', 1)
            mode = st.radio(
                "Mode", options=[1, 0],
                format_func=lambda m: "Major" if m == 1 else "Minor",
                index=0 if _auto_mode == 1 else 1,
                help="Major sounds brighter; minor sounds darker.",
                key="mode_radio",
            )
        _auto_ts = st.session_state.get('auto_time_signature', 4)
        _ts_options = [3, 4, 5, 6, 7]
        _ts_val = int(_auto_ts) if int(_auto_ts) in _ts_options else 4
        time_signature = st.select_slider(
            "Time signature", options=_ts_options, value=_ts_val,
            help="Beats per bar. 4/4 is by far the most common in popular music.",
            key="time_sig_slider",
        )

        st.markdown('<div class="header-divider" style="margin:0.7em 0;"></div>', unsafe_allow_html=True)

        # Save slider values so Song Insights tab can read them
        st.session_state['si_danceability']     = danceability
        st.session_state['si_energy']           = energy
        st.session_state['si_valence']          = valence
        st.session_state['si_acousticness']     = acousticness
        st.session_state['si_speechiness']      = speechiness
        st.session_state['si_instrumentalness'] = instrumentalness
        st.session_state['si_liveness']         = liveness
        st.session_state['si_loudness']         = loudness
        st.session_state['si_tempo']            = tempo

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
                    st.session_state['last_prediction'] = result
                    st.session_state['last_features']   = payload
                    st.session_state['last_genre']      = track_genre

                except requests.exceptions.ReadTimeout:
                    st.warning("⏳ The backend is waking up from sleep. Please wait a few seconds and try again.")
                    result = None

                except requests.exceptions.ConnectTimeout:
                    st.warning("⏳ The backend is starting up. Please wait a few seconds and try again.")
                    result = None

                except requests.exceptions.ConnectionError:
                    st.error("❌ Backend not reachable. Is the Render service running?")
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
                    _val_color = "#E0E0E0" if st.session_state.dark_mode else "#1A1A2E"
                    _hdr_color = "#888" if st.session_state.dark_mode else "#555"
                    _rows = "".join(
                        f'<tr><td style="color:#1DB954;font-weight:600;padding:6px 12px;">{k}</td>'
                        f'<td style="color:{_val_color};font-weight:500;padding:6px 12px;">{v}</td></tr>'
                        for k, v in payload.items()
                    )
                    st.markdown(f"""
                    <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
                        <thead><tr>
                            <th style="color:{_hdr_color};text-transform:uppercase;font-size:0.75em;
                                letter-spacing:0.8px;padding:6px 12px;border-bottom:1px solid rgba(29,185,84,0.20);
                                text-align:left;">Feature</th>
                            <th style="color:{_hdr_color};text-transform:uppercase;font-size:0.75em;
                                letter-spacing:0.8px;padding:6px 12px;border-bottom:1px solid rgba(29,185,84,0.20);
                                text-align:left;">Value</th>
                        </tr></thead>
                        <tbody>{_rows}</tbody>
                    </table>
                    """, unsafe_allow_html=True)

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
                        <tr><th>Feature</th><th>⬇️ Low value</th><th>⬆️ High value</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>💃 Danceability</td><td>Hard to dance to</td><td>Easy to dance to</td></tr>
                        <tr><td>⚡ Energy</td><td>Calm, soft</td><td>Intense, loud</td></tr>
                        <tr><td>😊 Valence</td><td>Sad / angry</td><td>Happy / euphoric</td></tr>
                        <tr><td>🎸 Acousticness</td><td>Electronic / synthetic</td><td>Acoustic instruments</td></tr>
                        <tr><td>🎙️ Speechiness</td><td>Pure music</td><td>Mostly spoken words</td></tr>
                        <tr><td>🎹 Instrumentalness</td><td>Has vocals</td><td>No vocals</td></tr>
                        <tr><td>🎤 Liveness</td><td>Studio recording</td><td>Live audience</td></tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SONG INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insights:

    _df = _load_dataset()

    col_left, col_right = st.columns([1.2, 1])

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    with col_left:

        # ── SECTION 1 — Songs like yours ─────────────────────────────────────
        st.markdown("### 🎵 Songs with a similar audio profile")
        st.caption("Based on the slider values you set in the Predictor tab")

        _FEATURES = ['danceability', 'energy', 'valence', 'acousticness',
                     'speechiness', 'instrumentalness', 'liveness', 'tempo']
        _SI_KEYS  = ['si_danceability', 'si_energy', 'si_valence', 'si_acousticness',
                     'si_speechiness', 'si_instrumentalness', 'si_liveness', 'si_tempo']

        if all(k in st.session_state for k in _SI_KEYS):
            from sklearn.preprocessing import MinMaxScaler

            _user_vec_raw = [[
                st.session_state['si_danceability'],
                st.session_state['si_energy'],
                st.session_state['si_valence'],
                st.session_state['si_acousticness'],
                st.session_state['si_speechiness'],
                st.session_state['si_instrumentalness'],
                st.session_state['si_liveness'],
                st.session_state['si_tempo'],
            ]]

            _scaler  = MinMaxScaler()
            _scaled  = _scaler.fit_transform(_df[_FEATURES])
            _uvec    = _scaler.transform(_user_vec_raw)
            _dists   = np.linalg.norm(_scaled - _uvec, axis=1)
            _df_sim  = _df.copy()
            _df_sim['similarity'] = 1 - (_dists / _dists.max())
            _top5    = _df_sim.nlargest(5, 'similarity')[
                ['track_name', 'artists', 'track_genre', 'popularity', 'similarity']
            ]

            for _, row in _top5.iterrows():
                _pop = int(row['popularity'])
                _sim_pct = int(round(row['similarity'] * 100))
                _track = str(row['track_name'])[:48]
                _artist = str(row['artists'])[:32]
                _genre  = str(row['track_genre'])

                if _pop >= 70:
                    _pop_html = f'<span class="pop-score-green">⬤ {_pop}</span>'
                elif _pop >= 50:
                    _pop_html = f'<span class="pop-score-orange">⬤ {_pop}</span>'
                else:
                    _pop_html = f'<span class="pop-score-gray">⬤ {_pop}</span>'

                if _sim_pct > 85:
                    _badge = f'<span class="match-badge-green">{_sim_pct}% match</span>'
                elif _sim_pct >= 70:
                    _badge = f'<span class="match-badge-blue">{_sim_pct}% match</span>'
                else:
                    _badge = f'<span class="match-badge-gray">{_sim_pct}% match</span>'

                st.markdown(f"""
                <div class="song-card">
                    <div class="song-card-info">
                        <div class="song-title">{_track}</div>
                        <div class="song-meta">{_artist} &nbsp;·&nbsp; {_genre} &nbsp;·&nbsp; Popularity: {_pop_html}</div>
                    </div>
                    {_badge}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon">🎚️</div>
                <div class="placeholder-text">
                    Set your audio features in the <strong style="color:#1DB954;">Predictor</strong> tab first,
                    then come back here.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🎭 Find your mood song")
        st.caption("Answer 3 quick questions and we'll find a real song from the dataset that matches your current vibe")

        # Animated intro
        st.markdown("""
<div style="background: linear-gradient(135deg, rgba(29,185,84,0.08), rgba(29,185,84,0.03));
border: 1px solid rgba(29,185,84,0.25); border-radius: 12px;
padding: 1.2rem; margin-bottom: 1.5rem; text-align: center;">
<div style="font-size: 2rem; margin-bottom: 0.5rem;">🎵 ✨ 🎶</div>
<p style="color: #888; margin: 0; font-size: 0.9em;">
Tell us how you feel right now — we'll find the perfect song for this moment
</p>
</div>
""", unsafe_allow_html=True)

        # 3 questions in columns
        q1, q2, q3 = st.columns(3)

        with q1:
            st.markdown("**😊 How do you feel?**")
            mood = st.select_slider(
                "Mood",
                options=["😢 Sad", "😐 Neutral", "🙂 Good", "😄 Happy", "🤩 Euphoric"],
                value="🙂 Good",
                label_visibility="collapsed"
            )

        with q2:
            st.markdown("**⚡ Energy level?**")
            energy_mood = st.select_slider(
                "Energy",
                options=["😴 Exhausted", "🧘 Calm", "🚶 Moderate", "🏃 Energetic", "🔥 Pumped"],
                value="🚶 Moderate",
                label_visibility="collapsed"
            )

        with q3:
            st.markdown("**🎯 What are you doing?**")
            activity = st.selectbox(
                "Activity",
                options=[
                    "🚗 Driving",
                    "📚 Studying",
                    "🏋️ Working out",
                    "😴 Relaxing",
                    "🎉 Partying",
                    "💔 Heartbreak",
                    "☕ Morning coffee",
                    "🌙 Late night",
                    "🧹 Doing chores",
                    "👯 Hanging with friends"
                ],
                label_visibility="collapsed"
            )

        # Map answers to audio feature ranges
        mood_map = {
            "😢 Sad":      {"valence": (0.0, 0.3), "energy": (0.0, 0.4)},
            "😐 Neutral":  {"valence": (0.3, 0.5), "energy": (0.3, 0.6)},
            "🙂 Good":     {"valence": (0.4, 0.7), "energy": (0.4, 0.7)},
            "😄 Happy":    {"valence": (0.6, 0.9), "energy": (0.5, 0.8)},
            "🤩 Euphoric": {"valence": (0.8, 1.0), "energy": (0.7, 1.0)},
        }

        energy_map = {
            "😴 Exhausted": {"energy": (0.0, 0.25), "tempo": (0, 80)},
            "🧘 Calm":      {"energy": (0.1, 0.4),  "tempo": (60, 100)},
            "🚶 Moderate":  {"energy": (0.35, 0.65),"tempo": (90, 120)},
            "🏃 Energetic": {"energy": (0.6, 0.85), "tempo": (110, 150)},
            "🔥 Pumped":    {"energy": (0.8, 1.0),  "tempo": (130, 220)},
        }

        activity_map = {
            "🚗 Driving":             {"danceability": (0.5, 0.9), "valence": (0.4, 0.9), "acousticness": (0.0, 0.5)},
            "📚 Studying":            {"instrumentalness": (0.3, 1.0), "energy": (0.0, 0.5), "speechiness": (0.0, 0.1)},
            "🏋️ Working out":         {"energy": (0.7, 1.0), "danceability": (0.6, 1.0), "tempo": (120, 220)},
            "😴 Relaxing":            {"energy": (0.0, 0.4), "acousticness": (0.3, 1.0), "valence": (0.3, 0.8)},
            "🎉 Partying":            {"danceability": (0.7, 1.0), "energy": (0.7, 1.0), "valence": (0.5, 1.0)},
            "💔 Heartbreak":          {"valence": (0.0, 0.3), "acousticness": (0.3, 1.0), "energy": (0.0, 0.5)},
            "☕ Morning coffee":      {"energy": (0.2, 0.6), "acousticness": (0.2, 0.8), "valence": (0.3, 0.7)},
            "🌙 Late night":          {"energy": (0.0, 0.5), "valence": (0.2, 0.6), "acousticness": (0.2, 0.9)},
            "🧹 Doing chores":        {"danceability": (0.5, 0.9), "energy": (0.4, 0.8), "valence": (0.4, 0.9)},
            "👯 Hanging with friends":{"danceability": (0.6, 1.0), "valence": (0.5, 1.0), "energy": (0.5, 0.9)},
        }

        find_btn = st.button("🎵 Find my mood song", use_container_width=True, type="primary")

        if find_btn:
            _df_mood = _load_dataset()

            # Build filter based on all 3 answers
            mask = pd.Series([True] * len(_df_mood))

            m = mood_map[mood]
            e = energy_map[energy_mood]
            a = activity_map[activity]

            # Apply valence from mood
            mask &= (_df_mood["valence"] >= m["valence"][0]) & (_df_mood["valence"] <= m["valence"][1])

            # Apply energy (combine mood and energy level)
            energy_min = max(m["energy"][0], e["energy"][0])
            energy_max = min(m["energy"][1], e["energy"][1])
            if energy_min < energy_max:
                mask &= (_df_mood["energy"] >= energy_min) & (_df_mood["energy"] <= energy_max)

            # Apply tempo from energy level
            if "tempo" in e:
                mask &= (_df_mood["tempo"] >= e["tempo"][0]) & (_df_mood["tempo"] <= e["tempo"][1])

            # Apply activity filters
            for feature, (fmin, fmax) in a.items():
                if feature in _df_mood.columns and feature != "tempo":
                    mask &= (_df_mood[feature] >= fmin) & (_df_mood[feature] <= fmax)

            filtered = _df_mood[mask].copy()

            # If too few results, relax filters and just use mood
            if len(filtered) < 10:
                filtered = _df_mood[
                    (_df_mood["valence"] >= m["valence"][0]) &
                    (_df_mood["valence"] <= m["valence"][1])
                ].copy()

            if len(filtered) == 0:
                st.warning("No songs found for this mood combination. Try different settings!")
            else:
                # Pick top 3 by popularity from filtered results
                top3 = filtered.nlargest(50, "popularity").sample(min(3, len(filtered)))

                st.markdown(f"""
                <div style="background: rgba(29,185,84,0.08); border: 1px solid rgba(29,185,84,0.25);
                border-radius: 10px; padding: 1rem; margin: 1rem 0; text-align: center;">
                <p style="color: #1DB954; font-weight: 600; margin: 0; font-size: 1.1em;">
                🎵 We found {len(filtered):,} songs that match your vibe!
                </p>
                <p style="color: #888; margin: 0.3rem 0 0; font-size: 0.85em;">
                Here are 3 top picks for: <b>{mood}</b> · <b>{energy_mood}</b> · <b>{activity}</b>
                </p>
                </div>
                """, unsafe_allow_html=True)

                for _, song in top3.iterrows():
                    pop_color = "#1DB954" if song["popularity"] >= 70 else "#FF9800" if song["popularity"] >= 50 else "#888"
                    song_query = f"{song['track_name']} {song['artists']}".replace(" ", "+")

                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
                    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem;">

                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="width: 44px; height: 44px; border-radius: 8px;
                        background: rgba(29,185,84,0.15); display: flex; align-items: center;
                        justify-content: center; font-size: 1.4rem; flex-shrink: 0;">🎵</div>
                        <div>
                            <div style="font-weight: 600; color: var(--text-primary); font-size: 1em;">
                            {song['track_name']}</div>
                            <div style="color: #888; font-size: 0.85em;">{song['artists']} · {song['track_genre']}</div>
                            <div style="font-size: 0.78em; color: #666; margin-top: 2px;">
                            valence {song['valence']:.2f} · energy {song['energy']:.2f} · tempo {song['tempo']:.0f} BPM
                            </div>
                        </div>
                    </div>

                    <div style="display: flex; align-items: center; gap: 1rem; flex-shrink: 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: {pop_color};">
                            {int(song['popularity'])}</div>
                            <div style="font-size: 0.7em; color: #666;">popularity</div>
                        </div>
                        <a href="https://www.youtube.com/results?search_query={song_query}"
                        target="_blank" style="background: #FF0000; color: white; padding: 0.4rem 0.9rem;
                        border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;
                        white-space: nowrap;">▶️ YouTube</a>
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── RIGHT COLUMN — Model Leaderboard ─────────────────────────────────────
    with col_right:

        st.markdown("### 🏆 Model Leaderboard")
        st.caption("All models compared on the held-out test set")

        st.markdown("**Classification Task** — Can we predict if a song is popular? (popularity ≥ 70)")

        st.markdown("""
<table style="width:100%; border-collapse: collapse; font-size: 0.85em; margin-bottom: 1rem;">
<thead>
<tr style="background: rgba(29,185,84,0.15); color: #1DB954;">
    <th style="padding: 8px; text-align: left; border-bottom: 1px solid rgba(29,185,84,0.3);">Model</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">Accuracy</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">F1 Score</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">ROC-AUC</th>
</tr>
</thead>
<tbody>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">1. Random Guessing</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">95.2%</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.000</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.500</td>
</tr>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">2. Logistic Regression</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">95.2%</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.141</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.706</td>
</tr>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">3. Decision Tree</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">95.3%</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.143</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.708</td>
</tr>
<tr style="background: rgba(29,185,84,0.12); color: #1DB954; font-weight: bold;">
    <td style="padding: 8px;">🏆 4. XGBoost (tuned)</td>
    <td style="padding: 8px; text-align: center;">95.8%</td>
    <td style="padding: 8px; text-align: center;">0.437</td>
    <td style="padding: 8px; text-align: center;">0.920</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)

        st.warning("⚠️ Accuracy is misleading here — F1 and ROC-AUC are the honest metrics.")

        st.markdown("**Regression Task** — Predict exact popularity score (0–100)")

        st.markdown("""
<table style="width:100%; border-collapse: collapse; font-size: 0.85em; margin-bottom: 1rem;">
<thead>
<tr style="background: rgba(29,185,84,0.15); color: #1DB954;">
    <th style="padding: 8px; text-align: left; border-bottom: 1px solid rgba(29,185,84,0.3);">Model</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">R²</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">MAE</th>
    <th style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(29,185,84,0.3);">RMSE</th>
</tr>
</thead>
<tbody>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">1. Random Guessing</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.000</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">18.5</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">22.1</td>
</tr>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">2. Ridge Regression</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.062</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">17.2</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">21.4</td>
</tr>
<tr style="color: #888;">
    <td style="padding: 8px; border-bottom: 1px solid rgba(255,255,255,0.05);">3. Decision Tree</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">0.089</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">16.8</td>
    <td style="padding: 8px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.05);">21.1</td>
</tr>
<tr style="background: rgba(29,185,84,0.12); color: #1DB954; font-weight: bold;">
    <td style="padding: 8px;">🏆 4. XGBoost (tuned)</td>
    <td style="padding: 8px; text-align: center;">0.380</td>
    <td style="padding: 8px; text-align: center;">13.1</td>
    <td style="padding: 8px; text-align: center;">16.7</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)

        st.info("📊 R²=0.38 means the model explains 38% of popularity. The other 62% = artist fame, marketing, social media.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_explain:

    if 'last_prediction' not in st.session_state or 'last_features' not in st.session_state:
        st.info("👆 Go to the Predictor tab, click **Predict popularity** first — then come back here to see a full explanation of your result.")
    else:
        _exp_pred   = st.session_state['last_prediction']
        _exp_feat   = st.session_state['last_features']
        score       = float(_exp_pred.get('popularity_score', 0))
        confidence  = float(_exp_pred.get('confidence', 0))
        is_popular  = bool(_exp_pred.get('popular', False) or _exp_pred.get('is_popular', False))
        genre       = _exp_feat.get('track_genre', '')
        danceability = float(_exp_feat.get('danceability', 0))
        energy       = float(_exp_feat.get('energy', 0))
        loudness     = float(_exp_feat.get('loudness', 0))
        acousticness = float(_exp_feat.get('acousticness', 0))
        tempo        = float(_exp_feat.get('tempo', 0))

        # ── Visual diagram — Your prediction step by step ────────────────────
        st.markdown("### 🔍 Your prediction explained — step by step")

        st.markdown(f"""
<div style="background: rgba(29,185,84,0.08); border: 2px solid #1DB954;
border-radius: 10px; padding: 1.2rem; text-align: center; margin-bottom: 0.5rem;">
<h4 style="color: #1DB954; margin: 0;">🎵 Your song</h4>
<p style="margin: 0.5rem 0 0;">
<b>{genre}</b> · danceability {danceability} · energy {energy} ·
loudness {loudness} dB · tempo {tempo:.0f} BPM
</p>
</div>

<div style="text-align: center; font-size: 1.5rem; color: #1DB954; margin: 0.2rem 0;">↓</div>

<div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15);
border-radius: 10px; padding: 1.2rem; text-align: center; margin-bottom: 0.5rem;">
<h4 style="margin: 0;">🔄 Preparation step</h4>
<p style="color: #888; margin: 0.5rem 0 0; font-size: 0.9em;">
Before the models could read your values, two quick things happened automatically:<br><br>
<b>1.</b> The genre name <b>"{genre}"</b> was converted into a number —
because machine learning models only understand numbers, not words.<br><br>
<b>2.</b> All your values were rescaled to the same range —
so that tempo (which can be 120 BPM) doesn't unfairly overpower
danceability (which is between 0 and 1).<br><br>
Then both models received the same prepared values at the same time.
</p>
</div>

<div style="display: flex; gap: 1rem; margin: 0.5rem 0;">
<div style="flex: 1; text-align: center; color: #1DB954; font-size: 1em; font-weight: 500;">
↙ sent to Regression model
</div>
<div style="flex: 1; text-align: center; color: #1DB954; font-size: 1em; font-weight: 500;">
↘ sent to Classification model
</div>
</div>
""", unsafe_allow_html=True)

        col_reg, col_clf = st.columns(2)

        with col_reg:
            st.markdown(f"""
<div style="background: rgba(29,185,84,0.08); border: 1px solid rgba(29,185,84,0.4);
border-radius: 10px; padding: 1.2rem;">

<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;">
    <span style="font-size: 1.5rem;">📈</span>
    <div>
        <h4 style="color: #1DB954; margin: 0;">Regression</h4>
        <p style="color: #888; font-size: 0.8em; margin: 0;">Predicts an exact number</p>
    </div>
</div>

<div style="display: flex; flex-direction: column; gap: 0.6rem;">
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">QUESTION</span><br>
        <span style="font-size: 0.9em;">What exact score will this song get?</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">TRAINED IN</span><br>
        <span style="font-size: 0.9em;">train.py — offline, using 91,200 songs</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">PREDICTION RUNS IN</span><br>
        <span style="font-size: 0.9em;">api.py — /predict function</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">SHOWN IN APP</span><br>
        <span style="font-size: 0.9em;">Predictor tab → green score bar</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">HOW DID THE MODEL GET {score:.1f}?</span><br>
        <span style="font-size: 0.9em;">
        During training, the model studied 91,200 songs and learned rules like:
        <i>"party songs with high energy and low acousticness tend to score around 55,
        while acoustic songs tend to score around 42."</i>
        When you clicked Predict, it applied those learned rules to your song
        and calculated <b style="color:#1DB954;">{score:.1f}</b> as the most likely score.
        Think of it like a teacher who graded 91,200 exams and now knows exactly
        what a {score:.0f}-point answer looks like.
        </span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">WHAT DOES {score:.1f} MEAN IN REAL LIFE?</span><br>
        <span style="font-size: 0.9em;">
        Spotify popularity scores go from 0 to 100.
        A score of <b style="color:#1DB954;">{score:.1f}</b> means this song would likely
        be {'a moderately known song — heard by some but not mainstream' if 40 <= score < 60 else 'a niche song with a small but dedicated audience' if score < 40 else 'a popular mainstream hit'}.
        For comparison: a viral hit like Blinding Lights scores ~87,
        an average song scores ~42, and unknown tracks score below 20.
        </span>
    </div>
</div>

<div style="background: rgba(29,185,84,0.15); border-radius: 8px;
padding: 0.8rem; text-align: center; margin-top: 1rem;">
    <span style="font-size: 2rem; font-weight: 900; color: #1DB954;">{score:.1f}</span>
    <span style="color: #888;"> / 100</span>
    <div style="color: #888; font-size: 0.78em; margin-top: 0.2rem;">MAE ±13.1 pts · R²=0.38</div>
</div>
</div>
""", unsafe_allow_html=True)

        with col_clf:
            clf_color   = "#1DB954" if is_popular else "#FF5252"
            clf_bg      = "rgba(29,185,84,0.08)" if is_popular else "rgba(255,82,82,0.08)"
            clf_border  = "rgba(29,185,84,0.4)"  if is_popular else "rgba(255,82,82,0.4)"
            clf_label   = "✅ Popular" if is_popular else "❌ Not Popular"
            clf_decision = "above" if is_popular else "below"

            st.markdown(f"""
<div style="background: {clf_bg}; border: 1px solid {clf_border};
border-radius: 10px; padding: 1.2rem;">

<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;">
    <span style="font-size: 1.5rem;">🎯</span>
    <div>
        <h4 style="color: {clf_color}; margin: 0;">Classification</h4>
        <p style="color: #888; font-size: 0.8em; margin: 0;">Puts song into a category</p>
    </div>
</div>

<div style="display: flex; flex-direction: column; gap: 0.6rem;">
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">QUESTION</span><br>
        <span style="font-size: 0.9em;">Is this song popular or not? (threshold: score ≥ 70)</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">TRAINED IN</span><br>
        <span style="font-size: 0.9em;">train.py — only 4.8% of songs were Popular</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">PREDICTION RUNS IN</span><br>
        <span style="font-size: 0.9em;">api.py — same /predict function</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">SHOWN IN APP</span><br>
        <span style="font-size: 0.9em;">Predictor tab → big label at the top</span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">HOW WAS THE DECISION MADE?</span><br>
        <span style="font-size: 0.9em;">
        The classifier learned that only 4.8% of all Spotify songs are truly popular
        (score ≥ 70). It gave your song a <b style="color:{'#1DB954' if is_popular else '#FF5252'};">{confidence*100:.1f}%</b> chance
        of being in that rare group. We use 50% as the decision line —
        if the model is more than 50% sure, it says Popular.
        Your song scored {confidence*100:.1f}% which is {clf_decision} that line,
        so the final answer is <b>{'Popular ✅' if is_popular else 'Not Popular ❌'}</b>.
        </span>
    </div>
    <div style="background: rgba(255,255,255,0.05); border-radius: 6px; padding: 0.6rem 0.8rem;">
        <span style="color: #888; font-size: 0.75em;">WHAT DOES THIS MEAN IN REAL LIFE?</span><br>
        <span style="font-size: 0.9em;">
        {'This song has the audio DNA of a commercial hit — high danceability, energy, and the right genre profile. Songs like this tend to get playlist placements and radio play.' if is_popular else 'This song does not match the audio patterns of mainstream popular songs. That does not mean it is a bad song — it means the audio features alone suggest it will not reach the Spotify mainstream. Many great songs stay niche.'}
        </span>
    </div>
</div>

<div style="background: {clf_bg}; border-radius: 8px; border: 1px solid {clf_border};
padding: 0.8rem; text-align: center; margin-top: 1rem;">
    <span style="font-size: 1.5rem; font-weight: 900; color: {clf_color};">{clf_label}</span>
    <div style="color: #888; font-size: 0.78em; margin-top: 0.2rem;">Confidence: {confidence*100:.1f}% · ROC-AUC=0.920</div>
</div>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── Model strengths and limitations ───────────────────────────────────
        st.markdown("### ⚖️ Model strengths and limitations")

        col_good, col_bad = st.columns(2)

        with col_good:
            st.markdown("""
<div style="background: rgba(29,185,84,0.08); border: 1px solid rgba(29,185,84,0.3);
border-radius: 10px; padding: 1.2rem;">
<h4 style="color: #1DB954; margin-top: 0;">✅ What this model does well</h4>

- Trained on 114,000 real Spotify songs — large and diverse dataset
- XGBoost handles class imbalance better than simpler models
- ROC-AUC of 0.920 — far better than random guessing (0.500)
- Predicts score within ±13 points on average
- Works across 114 different music genres
- Gives two answers at once: exact score + popular/not popular label
- Runs in under 1 second per prediction
</div>
""", unsafe_allow_html=True)

        with col_bad:
            st.markdown("""
<div style="background: rgba(255,82,82,0.08); border: 1px solid rgba(255,82,82,0.3);
border-radius: 10px; padding: 1.2rem;">
<h4 style="color: #FF5252; margin-top: 0;">❌ What this model cannot do</h4>

- Cannot see artist fame — a Taylor Swift song always scores higher
- Cannot measure marketing budget or label support
- Cannot predict TikTok or social media virality
- Only explains 38% of popularity — 62% is invisible to audio features
- Dataset from 2022 — music trends have changed since then
- Rare popular songs (4.8%) are harder to detect — F1 only 0.437
- Cannot account for release timing (summer hits, Christmas songs)
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EDA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:

    st.markdown("""
    <div class="eda-intro">
        <strong>What is EDA?</strong> Exploratory Data Analysis is the step we do
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
            st.image(path, width=700)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="eda-caption">{caption}</div>', unsafe_allow_html=True)
        else:
            st.info(
                f"**{filename}** not found.  \n"
                "Run `python train.py` to generate EDA plots."
            )

    # ── Section 1: Target variable ────────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Target Variable — Popularity</div>', unsafe_allow_html=True)
    st.markdown("""
**What am I looking at?** Two charts about song popularity scores (0–100).

The left chart shows how popular songs generally are — most songs score between 0 and 60,
meaning they are not very well known. Very few songs reach a score of 70 or higher.

The right chart shows the split we use for our model: only 1 in 20 songs (4.8%)
is considered "popular" (score ≥ 70). This means our dataset is heavily imbalanced —
like trying to find needles in a haystack. This is important because a model that
always guesses "not popular" would be right 95% of the time — but completely useless!
""")
    _show_plot("01_popularity_distribution.png", "")

    # ── Section 2: Correlations ───────────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Correlations</div>', unsafe_allow_html=True)
    st.markdown("""
**What am I looking at?** A grid showing how strongly each audio feature is connected to popularity.

Red squares = when one goes up, the other tends to go up too.
Blue squares = when one goes up, the other tends to go down.
The darker the color, the stronger the connection.

The most important row is the bottom one — it shows which features are most connected
to popularity. Loudness (how loud the song is) has the strongest positive connection.
Acousticness (how acoustic the song sounds) and instrumentalness (no vocals)
are negatively connected — meaning quieter, acoustic songs tend to be less popular.

Key insight: no single feature strongly predicts popularity on its own.
This is why we need a powerful model like XGBoost that combines all features together.
""")
    _show_plot("03_correlation_heatmap.png", "")

    # ── Section 3: Genre popularity ───────────────────────────────────────────
    st.markdown('<div class="eda-section-header">Genre Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
**What am I looking at?** A ranking of the top 25 music genres by their average popularity score.

The longer the bar, the more popular that genre is on average. Pop-film (music from movies),
k-pop, and chill music top the list. Heavy metal, death metal, and classical
genres tend to score much lower.

Key insight: genre is the single most powerful predictor of popularity in our dataset.
Knowing what genre a song belongs to tells us more about its likely popularity
than any audio feature like tempo or energy. This is why genre is included as
a feature in our prediction model.
""")
    _show_plot("06_genre_popularity.png", "")

    # ── Section 4: Model evaluation plots ────────────────────────────────────
    st.markdown('<div class="eda-section-header">Model Evaluation — XGBoost (Best Model)</div>', unsafe_allow_html=True)
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("""
**What am I looking at?** A table showing how well our model classified 22,800 songs as popular or not popular.

Think of it like a report card with 4 boxes:
- ✅ Top-left: songs that ARE popular and our model said YES — correct!
- ❌ Top-right: songs that ARE popular but our model said NO — missed them
- ❌ Bottom-left: songs that are NOT popular but our model said YES — false alarm
- ✅ Bottom-right: songs that are NOT popular and our model said NO — correct!

Our model is quite careful — it rarely gives false alarms, but it does miss
some popular songs. Overall it scores F1 = 0.437 and AUC = 0.920,
which is much better than random guessing (AUC = 0.500).
""")
        _show_plot("07_confusion_matrix_xgb.png", "")
    with c_right:
        st.markdown("""
**What am I looking at?** Each dot is one song from our test set.
The horizontal axis shows what score the song actually has.
The vertical axis shows what score our model predicted.

If our model were perfect, every dot would sit exactly on the red diagonal line.
Dots above the line = model overestimated. Dots below = model underestimated.

The spread of dots shows that our model gets the general direction right
but is not perfect — especially for very popular songs (80–100)
which are rare and hard to predict. R² = 0.38 means our model explains
38% of what makes a song popular. The other 62% comes from things
we cannot measure from audio alone — like how famous the artist is,
how much the label spent on marketing, or whether the song went viral on TikTok.
""")
        _show_plot("09_predicted_vs_actual_xgb.png", "")

    # ── Section 8: Model Leaderboard ──────────────────────────────────────────
    st.markdown("### 🏆 Model Leaderboard — All Models Compared")
    st.caption("Honest comparison of all 4 models on the held-out test set. Baseline (random guessing) is included so we can see how much each model improves.")

    st.markdown("**Classification Task** — Can we predict if a song is popular? (popularity ≥ 70)")

    def _leaderboard_table(headers, rows, last_row_green=True):
        th_style = (
            'background:rgba(29,185,84,0.12);color:#1DB954;font-weight:700;'
            'padding:9px 14px;border-bottom:1px solid rgba(29,185,84,0.25);'
            'text-align:left;font-size:0.82em;text-transform:uppercase;letter-spacing:0.6px;'
        )
        td_style = 'color:#C0C0C0;padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:0.88em;'
        td_last  = 'color:#1DB954;font-weight:600;padding:8px 14px;background:rgba(29,185,84,0.06);font-size:0.88em;'
        header_html = ''.join(f'<th style="{th_style}">{h}</th>' for h in headers)
        body_html = ''
        for i, row in enumerate(rows):
            is_last = last_row_green and i == len(rows) - 1
            cells = ''.join(f'<td style="{td_last if is_last else td_style}">{cell}</td>' for cell in row)
            body_html += f'<tr>{cells}</tr>'
        st.markdown(
            f'<div style="border:1px solid rgba(29,185,84,0.25);border-radius:10px;overflow:hidden;margin:0.5em 0;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    _leaderboard_table(
        headers=["Model", "Accuracy", "F1 Score", "ROC-AUC", "Notes"],
        rows=[
            ["1. Random Guessing (Baseline)", "95.2%", "0.000", "0.500", "Always predicts NOT popular — useless"],
            ["2. Logistic Regression",        "95.2%", "0.141", "0.706", "Linear model, limited by class imbalance"],
            ["3. Decision Tree (depth=6)",    "95.3%", "0.143", "0.708", "Slightly better, starts to overfit"],
            ["4. XGBoost (tuned) ✅ Best",    "95.8%", "0.437", "0.920", "Best overall — handles imbalance well"],
        ],
    )
    st.caption("⚠️ Note: Accuracy is misleading here — 95.2% of songs are NOT popular, so always guessing 'not popular' gives 95.2% accuracy. F1 Score and ROC-AUC are the honest metrics.")

    st.divider()

    st.markdown("**Regression Task** — Can we predict the exact popularity score (0–100)?")

    _leaderboard_table(
        headers=["Model", "R²", "MAE", "RMSE", "Notes"],
        rows=[
            ["1. Random Guessing (Baseline)", "0.000", "18.5", "22.1", "Always predicts mean score — baseline floor"],
            ["2. Ridge Regression",           "0.062", "17.2", "21.4", "Captures some linear signal"],
            ["3. Decision Tree (depth=6)",    "0.089", "16.8", "21.1", "Better but overfits on training data"],
            ["4. XGBoost (tuned) ✅ Best",    "0.380", "13.1", "16.7", "Best — explains 38% of score variance"],
        ],
    )
    st.caption("📊 R²=0.38 means our best model explains 38% of what makes a song popular. The remaining 62% comes from factors not in the audio data — marketing, artist fame, timing, social media.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:

    st.markdown('<div class="about-section">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-block">
        <span class="arch-phase">OFFLINE</span> &mdash; runs once on your machine<br>
        <span class="arch-sep">&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</span><br>
        dataset.csv <span class="arch-arrow">&rarr;</span> train.py <span class="arch-arrow">&rarr;</span> EDA plots + classifier.pkl + regressor.pkl<br><br>
        <span class="arch-phase">RUNTIME</span> &mdash; this app, always on<br>
        <span class="arch-sep">&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</span><br>
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

    # ── Live Model Stats ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📈 Live Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training samples", "91,200", help="Songs used to train the model")
    col2.metric("Test samples", "22,800", help="Songs used to evaluate the model")
    col3.metric("Best AUC score", "0.920", help="Area under ROC curve — 1.0 is perfect")
    col4.metric("Best R² score", "0.380", help="How much variance the model explains")

    # ── How to interpret results ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎯 How to Interpret the Prediction")
    st.markdown("""
| Score range | Meaning | What to do |
|-------------|---------|------------|
| 0 – 30 | Very unlikely to be popular | Reconsider arrangement or genre |
| 30 – 50 | Below average popularity | Some potential, needs work |
| 50 – 70 | Average popularity | Solid track, could reach mainstream |
| 70 – 85 | Likely popular ✅ | Strong commercial potential |
| 85 – 100 | Very likely popular 🔥 | Hit potential — prioritize release |
""")

    # ── Fun Facts ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🎵 Interesting Findings from 114,000 Songs")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("🎭 **Top genre**\n\npop-film scores highest average popularity across all genres")
    with col_b:
        st.info("📊 **Class imbalance**\n\nOnly 4.8% of songs score ≥70. Popularity is rare!")
    with col_c:
        st.info("🔊 **Loudness matters**\n\nLouder songs (closer to 0 dB) tend to score higher")
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        st.info("💃 **Dance to win**\n\nHigh danceability + high energy = stronger popularity signal")
    with col_e:
        st.info("🎸 **Genre is king**\n\nGenre alone explains more variance than any single audio feature")
    with col_f:
        st.info("⏱️ **Sweet spot duration**\n\n3–4 minute songs perform best on average")

    # ── Project Links ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔗 Project Links")
    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown("""
**📁 GitHub Repository**
All code, models, and data pipeline
[github.com/amilaherenda6/spotify-popularity-predictor](https://github.com/amilaherenda6/spotify-popularity-predictor)
""")
    with col_h:
        st.markdown(f"""
**🚀 API Documentation**
FastAPI backend with Swagger UI
[{API_URL}/docs]({API_URL}/docs)
""")

    # ── Song Insights Features ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Song Insights — Three New Features")

    with st.expander("🎵 Songs with a similar audio profile"):
        st.markdown("""
This feature takes the current slider values you set in the **Predictor** tab —
danceability, energy, valence, acousticness, speechiness, instrumentalness,
liveness, and tempo — and searches all **114,000 songs** in the Spotify dataset
to find the 5 closest matches.

**How similarity is calculated:**
1. All 8 features are normalized to a 0–1 scale using **MinMaxScaler** so that
   no single feature (e.g. tempo, which ranges up to 250) dominates the distance.
2. **Euclidean distance** is computed between your feature vector and every song
   in the dataset.
3. Distance is converted to a similarity score: `1 − (distance / max_distance)`,
   giving 100 % for a perfect match and lower percentages as songs diverge.

Each result card shows the **track name**, **artist**, **genre**, **popularity
score** (green ≥ 70, orange 50–69, gray < 50), and a **match percentage badge**
(green > 85 %, blue 70–85 %, gray otherwise).

This helps you understand what real songs your audio profile most resembles —
and what commercial popularity those songs actually achieved.
""")

    with st.expander("🎸 Genre audio profile explorer"):
        st.markdown("""
Select any of the **114 genres** in the dataset and instantly see the
**average audio fingerprint** of that genre — calculated across every song
tagged with that genre in the 114,000-track Spotify dataset.

**What is shown:**
- **Metric cards** for mean danceability, energy, valence, acousticness,
  average tempo (BPM), and average popularity score (out of 100).
- A **horizontal bar chart** of all 7 normalized features (0–1 scale) for
  easy visual comparison between genres.
- An **automatic insight message** that classifies the genre as
  high-popularity (avg > 55), moderate (avg 35–55), or niche (avg < 35),
  with a plain-language interpretation.

This helps you understand what audio characteristics define each genre and
benchmark how your own song's feature values compare to the genre average.
""")

    with st.expander("🚀 Song optimization suggestions"):
        st.markdown("""
After running a prediction in the **Predictor** tab, this section compares
your feature values against the **average values of popular songs**
(popularity ≥ 70) in the **same genre** from the dataset.

**How it works:**
1. All songs in your selected genre with popularity ≥ 70 are filtered from
   the dataset, and their mean feature values are computed.
2. The **absolute difference** between your values and those popular-song
   averages is calculated for each of 7 features (danceability, energy,
   valence, acousticness, speechiness, instrumentalness, liveness).
3. The **top 3 features** with the largest gap are surfaced as suggestions.

**Each suggestion card shows:**
- Feature name and direction (*increase* or *decrease*)
- Current value → target value (the popular-song average for that genre)
- Estimated point improvement: `difference × 15`, rounded, capped at **+15 per feature**
- A progress bar showing your current value on the 0–1 scale

At the bottom, a **total estimated new score** is displayed:
`current score + sum of improvements`, capped at **98** to remain realistic.

This gives you **actionable, data-driven advice** on which specific audio
characteristics to adjust to make your song more commercially competitive
within its genre.
""")

