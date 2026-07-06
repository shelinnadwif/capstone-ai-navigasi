"""
========================================================
app.py  —  Aplikasi Utama (Integrasi oleh Orang 4)
Capstone Project Kecerdasan Buatan - Kelompok 1
Universitas Mercu Buana — 2026

Struktur:
  Tab 1 : Navigasi Agen  (Orang 1 — tab_navigasi.py)
  Tab 2 : Mini-Game      (Orang 3/4 — tab_game_dan_modul.py)
  Tab 3 : Cuaca Bayes    (Orang 4 — tab_game_dan_modul.py)
  Tab 4 : Prediksi Energi(Orang 4 — tab_game_dan_modul.py)

Cara jalankan:
  pip install -r requirements.txt
  streamlit run app.py

Deploy:
  Streamlit Community Cloud  → streamlit.io/cloud
  Hugging Face Spaces        → huggingface.co/new-space
========================================================
"""

import streamlit as st

from tab_navigasi       import render_tab_navigasi
from tab_game_dan_modul import (
    render_tab_tictactoe,
    render_tab_cuaca,
    render_tab_energi,
)

# ── konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title = "Navigasi Agen Cerdas & Mini-Game Strategi",
    page_icon  = "🚚",
    layout     = "wide",
)

# ── inisialisasi session state global ───────────────────────────
st.session_state.setdefault("reached_goal",     False)
st.session_state.setdefault("last_path_length", 6)

# ── header ───────────────────────────────────────────────────────
st.title("🚚 Aplikasi Web Navigasi Agen Cerdas & Mini-Game Strategi")
st.caption(
    "Capstone Project — Mata Kuliah Kecerdasan Buatan - Kelompok 1 | "
    "Universitas Mercu Buana 2026"
)
st.markdown("---")

# ── tab utama ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Navigasi Agen (Grid 5×5)",
    "🎮 Mini-Game Tic-Tac-Toe",
    "☁️ Simulasi Cuaca (Bayes)",
    "🔋 Prediksi Energi (ML)",
])

with tab1:
    render_tab_navigasi()

with tab2:
    render_tab_tictactoe()

with tab3:
    render_tab_cuaca()

with tab4:
    render_tab_energi()
