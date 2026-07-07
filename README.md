# 🚚 Smart Agent Navigation & Strategy Mini-Game

> **Capstone Project — Mata Kuliah Kecerdasan Buatan**  
> **Kelompok 1 | Amellin ● Andre ● Noval ● Shelinna**
> **Universitas Mercu Buana — 2026**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

Aplikasi Web interaktif yang mensimulasikan perjalanan agen cerdas dari Gudang (Start) menuju Konsumen (Goal) di tengah rintangan. Proyek ini mengintegrasikan berbagai konsep **Kecerdasan Buatan (AI)** mulai dari *Uninformed/Informed Search*, *Adversarial Search*, *Probabilistic Reasoning*, hingga *Machine Learning*.

---

## ✨ Fitur Utama & Konsep AI

Aplikasi ini terdiri dari 4 Tab utama yang saling terintegrasi:

### 🗺️ 1. Navigasi Agen (Grid 5×5)
* **Konsep AI:** *State Space Representation*, *Uninformed Search* (BFS), *Informed Search* (A* dengan Manhattan Distance).
* **Fitur:** Visualisasi grid interaktif, perbandingan performa *node* yang dieksplorasi antara BFS dan A*, serta definisi formal 5 komponen *State Space*.

### 🎮 2. Mini-Game Tic-Tac-Toe vs AI Bot
* **Konsep AI:** *Adversarial Search* (Minimax Algorithm) dengan *Alpha-Beta Pruning*.
* **Fitur:** **Terkunci secara sistem!** Mini-game ini hanya akan terbuka jika agen berhasil mencapai *Goal* pada Tab 1. Menampilkan telemetri *real-time* jumlah *node* yang dievaluasi dan *cabang yang dipangkas (pruning)*.

### ☁️ 3. Simulasi Cuaca (Probabilitas Bayes)
* **Konsep AI:** *Probabilistic Reasoning* (Teorema Bayes).
* **Fitur:** Menghitung probabilitas posterior kemacetan berdasarkan kondisi cuaca (Hujan/Cerah) dan *prior knowledge* domain untuk memberikan rekomendasi rute.

### 🔋 4. Prediksi Sisa Energi Agen
* **Konsep AI:** *Machine Learning* (Decision Tree Regressor).
* **Fitur:** Model ML yang dilatih untuk memprediksi persentase sisa energi agen berdasarkan panjang rute (A*) dan jumlah rintangan, membantu pengambilan keputusan logistik.

---

## 🏗️ Arsitektur & State Space

Proyek ini mendefinisikan masalah navigasi secara formal:
| Komponen | Definisi |
| :--- | :--- |
| **Initial State** | Posisi agen di Gudang (Start) pada grid 5×5. |
| **Actions** | `{Atas, Bawah, Kiri, Kanan}` |
| **Transition Model** | `Result(s, a)` → sel tujuan jika valid & bukan rintangan. |
| **Goal Test** | Posisi agen == Posisi Konsumen (Goal). |
| **Path Cost** | Setiap langkah bernilai 1. |

---

## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Backend / Logic:** Python 3
* **Machine Learning:** `scikit-learn` (Decision Tree Regressor)
* **Data Manipulation:** `pandas`, `numpy`
* **Algoritma:** BFS, A* Search, Minimax, Alpha-Beta Pruning, Bayesian Inference
