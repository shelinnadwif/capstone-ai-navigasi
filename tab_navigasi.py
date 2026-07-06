"""
========================================================
BAGIAN ORANG 1 - Tab Navigasi Agen
Capstone Project Kecerdasan Buatan

Tanggung jawab:
- Grid 5x5 (Start, Obstacles, Goal)
- Algoritma BFS (Pencarian Buta)
- Algoritma A* Search (Pencarian Berpetunjuk)
- Tabel perbandingan node BFS vs A*
- Definisi State Space (5 komponen)
========================================================
"""

import streamlit as st
import pandas as pd
from collections import deque
import heapq

GRID_SIZE = 5

# =====================================================
# DEFINISI STATE SPACE (5 KOMPONEN)
# =====================================================
# 1. Initial State  : Posisi awal agen di koordinat Gudang (Start) -> (row, col)
# 2. Actions        : {Atas, Bawah, Kiri, Kanan} -> gerak 1 sel
# 3. Transition     : Result(s,a) -> sel tujuan jika valid & bukan rintangan
# 4. Goal Test      : posisi agen == posisi Konsumen (Goal)
# 5. Path Cost      : setiap langkah bernilai 1, total = jumlah langkah

# =====================================================
# FUNGSI UTILITAS GRID
# =====================================================
def in_bounds(pos):
    """Cek apakah posisi masih dalam batas grid 5x5."""
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def get_neighbors(pos, obstacles):
    """
    Transition Model: kembalikan sel-sel tetangga yang valid.
    Aksi: Atas(-1,0), Bawah(+1,0), Kiri(0,-1), Kanan(0,+1)
    """
    r, c = pos
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Atas, Bawah, Kiri, Kanan
    neighbors = []
    for dr, dc in moves:
        nxt = (r + dr, c + dc)
        if in_bounds(nxt) and nxt not in obstacles:
            neighbors.append(nxt)
    return neighbors


def reconstruct_path(parent, start, goal):
    """Telusuri balik dari goal ke start untuk mendapatkan jalur."""
    if goal not in parent:
        return []  # Goal tidak dapat dicapai
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path if path[0] == start else []


# =====================================================
# ALGORITMA 1: BFS (PENCARIAN BUTA / UNINFORMED)
# =====================================================
def bfs_search(start, goal, obstacles):
    """
    Breadth-First Search - menjelajah semua arah secara merata
    berdasarkan urutan kedalaman tanpa heuristik.

    Returns:
        path (list): jalur dari start ke goal
        explored_count (int): jumlah node yang dieksplorasi
    """
    frontier = deque([start])       # Antrian (FIFO)
    visited = {start}               # Node yang sudah dikunjungi
    parent = {start: None}          # Lacak jalur
    explored_count = 0

    while frontier:
        current = frontier.popleft()
        explored_count += 1

        # Goal Test
        if current == goal:
            break

        # Eksplorasi tetangga (Transition Model)
        for nxt in get_neighbors(current, obstacles):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                frontier.append(nxt)

    path = reconstruct_path(parent, start, goal)
    return path, explored_count


# =====================================================
# ALGORITMA 2: A* SEARCH (PENCARIAN BERPETUNJUK / INFORMED)
# =====================================================
def manhattan_distance(pos_a, pos_b):
    """
    Fungsi Heuristik h(n) = |x_n - x_goal| + |y_n - y_goal|
    Cocok untuk grid karena agen hanya bergerak horizontal/vertikal.
    """
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def astar_search(start, goal, obstacles):
    """
    A* Search - menggabungkan cost aktual g(n) dan heuristik h(n)
    menjadi f(n) = g(n) + h(n) untuk mengarahkan pencarian.

    Returns:
        path (list): jalur dari start ke goal
        explored_count (int): jumlah node yang dieksplorasi
    """
    # Priority queue: (f(n), g(n), posisi)
    frontier = [(manhattan_distance(start, goal), 0, start)]
    parent = {start: None}
    cost_so_far = {start: 0}    # g(n) untuk setiap node
    visited = set()
    explored_count = 0

    while frontier:
        _, g, current = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        explored_count += 1

        # Goal Test
        if current == goal:
            break

        # Eksplorasi tetangga
        for nxt in get_neighbors(current, obstacles):
            new_cost = cost_so_far[current] + 1     # g(n+1) = g(n) + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                h = manhattan_distance(nxt, goal)   # h(n)
                f = new_cost + h                    # f(n) = g(n) + h(n)
                heapq.heappush(frontier, (f, new_cost, nxt))
                parent[nxt] = current

    path = reconstruct_path(parent, start, goal)
    return path, explored_count


# =====================================================
# TAMPILAN UI STREAMLIT - TAB NAVIGASI
# =====================================================
def render_tab_navigasi():
    """Render semua konten Tab Navigasi Agen di Streamlit."""

    st.subheader("🗺️ Navigasi Agen pada Grid 5×5")

    # --- Sidebar pengaturan grid ---
    st.markdown("#### Pengaturan Posisi")
    col_set, col_grid = st.columns([1, 1.2])

    with col_set:
        start_row = st.number_input("Baris Start (Gudang)", 0, 4, 0, key="sr")
        start_col = st.number_input("Kolom Start (Gudang)", 0, 4, 0, key="sc")
        goal_row  = st.number_input("Baris Goal (Konsumen)", 0, 4, 4, key="gr")
        goal_col  = st.number_input("Kolom Goal (Konsumen)", 0, 4, 4, key="gc")

        algo = st.radio(
            "Pilih Algoritma",
            ["BFS (Pencarian Buta)", "A* Search (Berpetunjuk)"],
            key="algo"
        )

        run_btn = st.button("▶ Jalankan Navigasi", type="primary", key="run_nav")

    start     = (start_row, start_col)
    goal      = (goal_row, goal_col)
    obstacles = {(1, 1), (1, 2), (2, 3), (3, 1)}  # Default rintangan

    # --- Jalankan pencarian ---
    if run_btn:
        if start in obstacles or goal in obstacles:
            st.error("❌ Posisi Start atau Goal tidak boleh berada di atas rintangan!")
            return

        path_bfs,   exp_bfs   = bfs_search(start, goal, obstacles)
        path_astar, exp_astar = astar_search(start, goal, obstacles)

        chosen_path = path_bfs if algo.startswith("BFS") else path_astar

        # --- Tampilkan Grid ---
        with col_grid:
            st.markdown(f"**Grid hasil {algo.split(' ')[0]}**")
            grid_html = "<div style='font-size:28px; line-height:1.5;'>"
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    cell = (r, c)
                    if cell == start:
                        grid_html += "🟩"   # Gudang (Start)
                    elif cell == goal:
                        grid_html += "🟥"   # Konsumen (Goal)
                    elif cell in obstacles:
                        grid_html += "⬛"   # Rintangan
                    elif cell in chosen_path:
                        grid_html += "🔵"   # Jalur agen
                    else:
                        grid_html += "⬜"   # Sel kosong
                grid_html += "<br>"
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)

            st.caption("🟩 Start   🟥 Goal   ⬛ Rintangan   🔵 Jalur   ⬜ Kosong")

        # --- Hasil navigasi ---
        if chosen_path:
            st.success(
                f"✅ Agen berhasil mencapai tujuan dalam "
                f"**{len(chosen_path) - 1} langkah**."
            )
            # Simpan status untuk mini-game (dipakai Orang 2)
            st.session_state["reached_goal"]     = True
            st.session_state["last_path_length"] = len(chosen_path) - 1
        else:
            st.warning("⚠️ Jalur tidak ditemukan — semua rute terhalang rintangan.")
            st.session_state["reached_goal"] = False

        # --- Tabel perbandingan (BUKTI CAPAIAN Sub-CPMK 6.1.1) ---
        st.markdown("---")
        st.markdown("#### 📊 Tabel Perbandingan: BFS vs A* Search")

        len_bfs   = len(path_bfs)   - 1 if path_bfs   else "-"
        len_astar = len(path_astar) - 1 if path_astar else "-"

        df_compare = pd.DataFrame({
            "Algoritma"              : ["BFS (Pencarian Buta)", "A* Search (Berpetunjuk)"],
            "Node Dieksplorasi"      : [exp_bfs, exp_astar],
            "Panjang Jalur (langkah)": [len_bfs, len_astar],
        })
        st.table(df_compare)

        selisih = exp_bfs - exp_astar
        if selisih > 0:
            st.info(
                f"💡 A\\* mengeksplorasi **{selisih} node lebih sedikit** dari BFS "
                f"karena heuristik Manhattan Distance mengarahkan pencarian ke posisi Goal, "
                f"sementara BFS mengeksplorasi semua arah secara merata."
            )

        if st.session_state.get("reached_goal"):
            st.info(
                "🎯 Agen tiba di Konsumen! Buka tab **Mini-Game Tic-Tac-Toe** "
                "untuk melanjutkan ke sesi kompetitif."
            )

    # --- Expander: Definisi State Space (BUKTI CAPAIAN Sub-CPMK 3.1.1) ---
    st.markdown("---")
    with st.expander("📄 Definisi Formal State Space (Klik untuk lihat)"):
        st.markdown("""
        | Komponen | Definisi |
        |---|---|
        | **Initial State** | Posisi awal agen pada koordinat Gudang (Start) di grid 5×5. |
        | **Actions** | Himpunan aksi `{Atas, Bawah, Kiri, Kanan}` — menggerakkan agen satu sel. |
        | **Transition Model** | `Result(s, a)` menghasilkan sel tujuan jika masih dalam batas grid dan bukan rintangan; jika tidak valid, posisi agen tidak berubah. |
        | **Goal Test** | Bernilai **benar** ketika posisi agen sama dengan koordinat Konsumen (Goal). |
        | **Path Cost** | Setiap perpindahan bernilai 1; total path cost = jumlah langkah dari Start ke Goal. |
        """)
