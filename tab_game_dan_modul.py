"""
========================================================
tab_game_dan_modul.py  —  BAGIAN ORANG 4
Capstone Project Kecerdasan Buatan

Berisi:
  A. Mini-Game Tic-Tac-Toe  (Minimax + Alpha-Beta Pruning)
     Logika diadaptasi dari P14/tic-tac-toe-minimax.html
     milik Orang 3, diterjemahkan dari JS ke Python/Streamlit.
  B. Simulasi Cuaca — Probabilitas Bayes Sederhana
  C. Prediksi Sisa Energi Agen — scikit-learn Decision Tree
========================================================
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeRegressor


# ================================================================
# BAGIAN A — MINI-GAME TIC-TAC-TOE
# Minimax + Alpha-Beta Pruning
# (diadaptasi dari tic-tac-toe-minimax.html — Orang 3 / P14)
# ================================================================

_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # baris
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # kolom
    [0, 4, 8], [2, 4, 6],               # diagonal
]


def check_winner(board):
    """
    Periksa kondisi terminal permainan.
    board : list[9] berisi 'X', 'O', atau ''.
    Return: 'X' | 'O' | 'Seri' | None (masih berlanjut).
    """
    for a, b, c in _LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if "" not in board:
        return "Seri"
    return None


def minimax(board, depth, alpha, beta, is_maximizing,
            stats: dict):
    """
    Algoritma Minimax dengan Alpha-Beta Pruning.

    AI Bot  = MAX player, simbol "O"  → skor makin tinggi makin baik
    Manusia = MIN player, simbol "X"  → skor makin rendah makin baik

    Alpha : nilai terbaik yang sudah dijamin untuk MAX
    Beta  : nilai terbaik yang sudah dijamin untuk MIN

    PRUNING terjadi saat beta <= alpha:
      – di node MAX  → cabang MIN di bawahnya tidak perlu dievaluasi
      – di node MIN  → cabang MAX di bawahnya tidak perlu dievaluasi
    """
    stats["nodes_evaluated"] += 1

    winner = check_winner(board)
    if winner == "O":
        return 10 - depth
    if winner == "X":
        return depth - 10
    if winner == "Seri":
        return 0

    if is_maximizing:
        best = -math.inf
        for i in range(9):
            if board[i] == "":
                board[i] = "O"
                val = minimax(board, depth + 1, alpha, beta, False, stats)
                board[i] = ""
                best  = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:
                    stats["branches_pruned"] += 1
                    break   # ← PRUNING
        return best
    else:
        best = math.inf
        for i in range(9):
            if board[i] == "":
                board[i] = "X"
                val = minimax(board, depth + 1, alpha, beta, True, stats)
                board[i] = ""
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:
                    stats["branches_pruned"] += 1
                    break   # ← PRUNING
        return best


def best_ai_move(board):
    """
    Pilih langkah terbaik AI. Return (index sel, stats dict).
    """
    stats = {"nodes_evaluated": 0, "branches_pruned": 0}
    best_score, best_move = -math.inf, None
    for i in range(9):
        if board[i] == "":
            board[i] = "O"
            score = minimax(board, 0, -math.inf, math.inf, False, stats)
            board[i] = ""
            if score > best_score:
                best_score, best_move = score, i
    return best_move, stats


def render_tab_tictactoe():
    """Render Tab Mini-Game Tic-Tac-Toe."""
    st.subheader("🎮 Mini-Game Tic-Tac-Toe vs AI Bot")

    if not st.session_state.get("reached_goal", False):
        st.warning(
            "⚠️ Mini-game terkunci. Selesaikan navigasi agen "
            "hingga mencapai Konsumen (Goal) pada tab **Navigasi** terlebih dahulu."
        )
        return

    st.success("🔓 Mini-game terbuka — agen berhasil mencapai tujuan!")
    st.markdown("Kamu bermain sebagai **X** · AI Bot bermain sebagai **O** · Kamu giliran pertama.")

    # inisialisasi state
    if "ttt_board"  not in st.session_state:
        st.session_state.ttt_board  = [""] * 9
    if "ttt_status" not in st.session_state:
        st.session_state.ttt_status = None
    if "ttt_stats"  not in st.session_state:
        st.session_state.ttt_stats  = {"nodes_evaluated": 0, "branches_pruned": 0}

    board  = st.session_state.ttt_board
    status = st.session_state.ttt_status

    # --- papan 3×3 ---
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            idx      = row * 3 + col
            label    = board[idx] if board[idx] else " "
            disabled = bool(board[idx]) or status is not None
            with cols[col]:
                if st.button(label, key=f"ttt_{idx}",
                             disabled=disabled, use_container_width=True):
                    board[idx] = "X"
                    winner = check_winner(board)
                    if winner:
                        st.session_state.ttt_status = winner
                    else:
                        move, stats = best_ai_move(board)
                        if move is not None:
                            board[move] = "O"
                            st.session_state.ttt_stats = stats
                            winner = check_winner(board)
                            if winner:
                                st.session_state.ttt_status = winner
                    st.rerun()

    # --- telemetri pruning (seperti di HTML Orang 3) ---
    s = st.session_state.ttt_stats
    c1, c2 = st.columns(2)
    c1.metric("Node Dievaluasi (giliran terakhir AI)", s["nodes_evaluated"])
    c2.metric("Cabang Dipangkas (pruning)",            s["branches_pruned"])

    # --- hasil ---
    if status:
        st.markdown("---")
        if status == "Seri":
            st.info("🤝 Permainan berakhir seri!")
        elif status == "X":
            st.success("🎉 Kamu (X) menang!")
        else:
            st.error("🤖 AI Bot (O) menang! Coba lagi.")
        if st.button("🔄 Main Lagi", key="reset_ttt"):
            st.session_state.ttt_board  = [""] * 9
            st.session_state.ttt_status = None
            st.session_state.ttt_stats  = {"nodes_evaluated": 0, "branches_pruned": 0}
            st.rerun()

    # --- analisis pruning (BUKTI CAPAIAN Sub-CPMK 6.2.1) ---
    st.markdown("---")
    with st.expander("📄 Analisis Alpha-Beta Pruning"):
        st.markdown("""
Proses pemangkasan (*pruning*) terjadi ketika nilai **beta** pada suatu
simpul MIN menjadi lebih kecil atau sama dengan nilai **alpha** milik
simpul MAX di atasnya — atau sebaliknya. Pada kondisi tersebut, cabang-cabang
anak yang belum dievaluasi pasti tidak akan pernah memengaruhi keputusan
akhir pemain di level yang lebih tinggi, sehingga evaluasinya dihentikan
lebih awal.

Dalam permainan Tic-Tac-Toe ini, pruning paling banyak terjadi pada
**langkah awal hingga pertengahan** (sel kosong 9–4) ketika *branching factor*
masih besar. Begitu AI Bot menemukan satu lintasan yang menjamin kemenangan
atau setidaknya hasil seri, simulasi lintasan-lintasan alternatif yang
nilainya pasti lebih buruk langsung dihentikan tanpa mengubah keputusan
akhir. Sebaliknya, pada langkah-langkah akhir (sel kosong 1–2), potensi
pruning sangat kecil karena hampir seluruh cabang memang harus dievaluasi
untuk memastikan hasilnya. Telemetri di atas menampilkan angka nyata dari
eksekusi Minimax giliran terakhir sebagai bukti empiris proses ini berjalan.
        """)


# ================================================================
# BAGIAN B — SIMULASI CUACA: PROBABILITAS BAYES SEDERHANA
# ================================================================

def hitung_bayes(cuaca: str, prior_hujan: float) -> dict:
    """
    Posterior probability kondisi cuaca terhadap kemacetan.

    Prior pengetahuan domain:
        P(Macet | Hujan)  = 0.75
        P(Macet | Cerah)  = 0.20

    Bayes: P(H|E) = P(E|H) × P(H) / P(E)
    """
    p_mh = 0.75          # P(Macet | Hujan)
    p_mc = 0.20          # P(Macet | Cerah)
    p_h  = prior_hujan
    p_c  = 1 - p_h

    p_macet = p_mh * p_h + p_mc * p_c   # evidence total

    if "Hujan" in cuaca:
        posterior = (p_mh * p_h) / p_macet
        rek = ("⚠️ Hindari rute rawan macet — probabilitas kemacetan tinggi."
               if posterior > 0.5
               else "✅ Rute masih dapat dilalui meski hujan.")
    else:
        posterior = (p_mc * p_c) / p_macet
        rek = ("✅ Cuaca cerah, rute aman dilalui."
               if posterior < 0.5
               else "⚠️ Waspadai kepadatan rute meski cuaca cerah.")

    return {"posterior": posterior, "p_macet": p_macet, "rekomendasi": rek}


def render_tab_cuaca():
    """Render Tab Simulasi Cuaca."""
    st.subheader("☁️ Simulasi Cuaca — Probabilitas Bayes Sederhana")

    col_in, col_out = st.columns(2)
    with col_in:
        cuaca = st.selectbox("Kondisi Cuaca", ["Cerah ☀️", "Hujan ☔"], key="cwx")
        prior = st.slider("P(Hujan) — Probabilitas Prior",
                          0.0, 1.0, 0.30, 0.05, key="prx")
        hitung_btn = st.button("🔍 Hitung Probabilitas", key="btn_bayes")

    if hitung_btn:
        h = hitung_bayes(cuaca, prior)
        with col_out:
            st.metric(f"P({cuaca.split()[0]} | Macet)",
                      f"{h['posterior']:.2%}")
            st.metric("P(Macet) total", f"{h['p_macet']:.2%}")
            st.markdown(f"**Rekomendasi:** {h['rekomendasi']}")

        st.caption(
            "Rumus: P(H|E) = P(E|H) × P(H) / P(E) — "
            "dengan P(Macet|Hujan)=0.75 dan P(Macet|Cerah)=0.20 "
            "sebagai pengetahuan domain awal."
        )


# ================================================================
# BAGIAN C — PREDIKSI SISA ENERGI: DECISION TREE REGRESSOR
# ================================================================

@st.cache_resource
def latih_model():
    rng = np.random.RandomState(42)
    n   = 300
    pj  = rng.randint(1, 20, n)
    ri  = rng.randint(0, 6,  n)
    nz  = rng.normal(0, 2,   n)
    y   = np.clip(100 - pj * 3.5 - ri * 2 + nz, 0, 100)
    X   = pd.DataFrame({"panjang_rute": pj, "jumlah_rintangan": ri})
    m   = DecisionTreeRegressor(max_depth=4, random_state=42)
    m.fit(X, y)
    return m


def prediksi_energi(model, panjang, rintangan):
    return round(float(np.clip(
        model.predict([[panjang, rintangan]])[0], 0, 100)), 1)


def render_tab_energi():
    """Render Tab Prediksi Energi."""
    st.subheader("🔋 Prediksi Sisa Energi Agen — Decision Tree Regressor")

    model   = latih_model()
    def_len = int(st.session_state.get("last_path_length", 6))

    col_in, col_out = st.columns(2)
    with col_in:
        panjang   = st.number_input("Panjang Rute A* (langkah)", 1, 30,
                                    def_len, key="inp_pj")
        rintangan = st.number_input("Estimasi Rintangan di Sekitar Rute", 0, 10,
                                    1, key="inp_ri")
        pred_btn  = st.button("⚡ Prediksi Energi", key="btn_ml")

    if pred_btn:
        hasil = prediksi_energi(model, panjang, rintangan)
        with col_out:
            st.metric("Estimasi Sisa Energi Agen", f"{hasil} %")
            if hasil >= 60:
                st.success(f"✅ Energi cukup ({hasil}%) — rute aman ditempuh.")
            elif hasil >= 30:
                st.warning(f"⚠️ Energi terbatas ({hasil}%) — pertimbangkan rute alternatif.")
            else:
                st.error(f"❌ Energi kritis ({hasil}%) — agen perlu pengisian daya.")
        st.caption(
            "Model: Decision Tree Regressor (scikit-learn, max_depth=4). "
            "Semakin panjang rute A* dan semakin banyak rintangan, "
            "semakin besar energi yang terpakai."
        )
