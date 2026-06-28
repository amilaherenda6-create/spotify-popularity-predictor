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
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg,#1DB954 0%,#17a349 100%);
    color: #fff; border: none; border-radius: 50px;
    font-size: 1.05em; font-weight: 700; letter-spacing: 0.8px;
    padding: 0.7em 2em; transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(29,185,84,0.35); text-transform: uppercase;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px); box-shadow: 0 6px 28px rgba(29,185,84,0.55);
}}
.stButton > button[kind="primary"]:active {{ transform: translateY(0); }}
.stButton > button[kind="primary"]:disabled {{
    background: {t['tab_hover']}; box-shadow: none; color: {t['text_vdim']};
}}

/* ── Theme toggle (secondary button) ────────────────────────────────────── */
.stButton > button[kind="secondary"] {{
    background: {t['toggle_bg']}; color: {t['toggle_text']};
    border: 1px solid {t['toggle_border']}; border-radius: 20px;
    font-size: 0.82em; font-weight: 600; padding: 5px 14px;
    transition: all 0.2s ease; letter-spacing: 0.3px;
}}
.stButton > button[kind="secondary"]:hover {{
    background: {t['toggle_hover']}; border-color: #1DB954; color: #1DB954;
}}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
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
[data-testid="stDataFrame"] {{
    border: 1px solid {t['border']}; border-radius: 8px; background: {t['expander_bg']};
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
</style>
"""


# Inject theme CSS
st.markdown(_build_css(_DARK if st.session_state.dark_mode else _LIGHT), unsafe_allow_html=True)

# =============================================================================
# HEADER — title left, theme toggle right
# =============================================================================
hdr_col, btn_col = st.columns([7, 1])

with hdr_col:
    st.markdown("""
    <div class="app-title">🎵 Spotify Popularity Predictor</div>
    <div class="app-tagline">Predict a song's commercial success from its audio DNA — powered by XGBoost</div>
    <div class="header-divider"></div>
    """, unsafe_allow_html=True)

with btn_col:
    st.markdown("<div style='padding-top:0.4em'></div>", unsafe_allow_html=True)
    toggle_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
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

    # ── Section 8: Model Leaderboard ──────────────────────────────────────────
    st.markdown("### 🏆 Model Leaderboard — All Models Compared")
    st.caption("Honest comparison of all 4 models on the held-out test set. Baseline (random guessing) is included so we can see how much each model improves.")

    st.markdown("**Classification Task** — Can we predict if a song is popular? (popularity ≥ 70)")

    clf_data = {
        "Model": [
            "1. Random Guessing (Baseline)",
            "2. Logistic Regression",
            "3. Decision Tree (depth=6)",
            "4. XGBoost (tuned) ✅ Best"
        ],
        "Accuracy": ["95.2%", "95.2%", "95.3%", "95.8%"],
        "F1 Score": ["0.000", "0.141", "0.143", "0.437"],
        "ROC-AUC": ["0.500", "0.706", "0.708", "0.920"],
        "Notes": [
            "Always predicts NOT popular -- useless",
            "Linear model, limited by class imbalance",
            "Slightly better, starts to overfit",
            "Best overall -- handles imbalance well"
        ]
    }

    clf_df = pd.DataFrame(clf_data)
    st.dataframe(clf_df, use_container_width=True, hide_index=True)

    st.caption("⚠️ Note: Accuracy is misleading here — 95.2% of songs are NOT popular, so always guessing 'not popular' gives 95.2% accuracy. F1 Score and ROC-AUC are the honest metrics.")

    st.divider()

    st.markdown("**Regression Task** — Can we predict the exact popularity score (0–100)?")

    reg_data = {
        "Model": [
            "1. Random Guessing (Baseline)",
            "2. Ridge Regression",
            "3. Decision Tree (depth=6)",
            "4. XGBoost (tuned) ✅ Best"
        ],
        "R²": ["0.000", "0.062", "0.089", "0.380"],
        "MAE": ["18.5", "17.2", "16.8", "13.1"],
        "RMSE": ["22.1", "21.4", "21.1", "16.7"],
        "Notes": [
            "Always predicts mean score -- baseline floor",
            "Captures some linear signal",
            "Better but overfits on training data",
            "Best -- explains 38% of score variance"
        ]
    }

    reg_df = pd.DataFrame(reg_data)
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

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
