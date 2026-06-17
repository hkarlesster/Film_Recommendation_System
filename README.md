# 🎬 CineMatch — Film Recommendation System with SVD

Aplikasi Web Sistem Rekomendasi Film berbasis AI yang dikembangkan sebagai **Proyek Akhir (Capstone Project) Pelatihan PIJAK**, sebuah program kolaborasi inklusif antara **Dicoding Indonesia** dan **IBM SkillsBuild** untuk transformasi digital siswa vokasi.

---

## 📌 Tentang Proyek
**CineMatch** dirancang untuk mengatasi tantangan *data sparsity* (keterbatasan data rating) pada platform streaming film. Sistem ini menggunakan teknik **Collaborative Filtering** dengan algoritma **Singular Value Decomposition (SVD)** untuk mempelajari pola preferensi tersembunyi (*latent factors*) dari pengguna. 

Aplikasi ini mengintegrasikan model machine learning yang telah dilatih secara offline (dieksport menggunakan `joblib`) ke dalam antarmuka interaktif berbasis **Streamlit web application**.

### 📊 Spesifikasi Proyek & Dataset
* **Dataset:** MovieLens Latest Small (100.836 rating dari 610 user untuk 9.742 film).
* **Algoritma:** Singular Value Decomposition (SVD).
* **Metrik Evaluasi:** Target RMSE $< 1.0$ dan Precision@10 $> 60\%$.
* **Tech Stack:** Python, Streamlit, Scikit-Surprise, Pandas, Joblib.

---

## 🎯 Fitur Utama Aplikasi (Streamlit)
1.  **🏠 Beranda:** Memuat ringkasan eksekutif proyek beserta visualisasi metrik performa database secara *real-time*.
2.  **🎯 Prediksi Rating:** Menguji estimasi skor kepuasan (skala 1-5) yang diproyeksikan akan diberikan oleh user tertentu terhadap suatu judul film. Lengkap dengan sistem rekomendasi berbasis *status badge*.
3.  **🍿 Rekomendasi Personal (Top-10):** Menghasilkan 10 rekomendasi film terbaik yang belum pernah ditonton oleh user, diurutkan berdasarkan prediksi matriks SVD tertinggi.

---

## 📂 Struktur Direktori Proyek
Pastikan struktur folder kamu terlihat seperti ini agar aplikasi berjalan lancar:

```text
📂 direktori-proyek/
│
├── app2.py                       # File utama aplikasi web Streamlit
├── copy_of_cinematch_svd_fix.py  # Script latihan model & EDA (Jupyter/Colab)
├── requirements.txt              # Daftar dependensi library Python
│
# --- File di bawah ini wajib berada di folder utama setelah training ---
├── movies.csv                    # File data film (diekstrak dari MovieLens)
└── cinematch_svd_model.pkl       # File model SVD hasil ekspor joblib
