import streamlit as st
import pandas as pd
import os
import joblib  # Menggunakan joblib sesuai kebutuhan Anda

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CineMatch - PIJAK AI Recommendation System", 
    page_icon="🎬", 
    layout="wide"
)

# --- CSS CUSTOM UNTUK TAMPILAN ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 6px; height: 3em; background-color: #1A73E8; color: white; font-weight: bold; }
    .stButton>button:hover { background-color: #1557B0; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI MEMUAT MODEL DENGAN JOBLIB ---
@st.cache_resource
def load_model_joblib(model_path):
    if os.path.exists(model_path):
        try:
            # joblib.load langsung mengembalikan objek model SVD Anda
            algo = joblib.load(model_path)
            return algo
        except Exception as e:
            st.error(f"Gagal membaca file menggunakan Joblib: {e}")
            st.info("💡 Solusi: Pastikan versi library 'scikit-surprise' dan 'joblib' di lingkungan ini sama dengan saat Anda melakukan training model.")
            return None
    else:
        return None

# --- NAMA FILE MODEL ---
MODEL_FILE = "cinematch_svd_model.pkl"
model = load_model_joblib(MODEL_FILE)

# --- SIDEBAR IDENTITAS PROGRAM ---
with st.sidebar:
    st.image("https://www.dicoding.com/blog/wp-content/uploads/2023/10/Logo-Dicoding-IBM-SkillsBuild.png", width=220)
    st.title("📌 Menu Navigasi")
    menu = st.radio("Pilih Halaman:", ["🏠 Beranda", "🎯 Prediksi Rating", "🍿 Rekomendasi Personal"])
    st.markdown("---")
    st.info("**Program PIJAK**\n\nKolaborasi Strategis Dicoding & IBM SkillsBuild untuk Transformasi Digital.")

# --- KONDISI JIKA MODEL TIDAK DITEMUKAN ---
if model is None:
    st.title("🎬 CineMatch - Sistem Rekomendasi")
    st.error(f"❌ File `{MODEL_FILE}` gagal dimuat atau tidak ditemukan!")
    st.warning(f"Pastikan file model bernama **`{MODEL_FILE}`** (hasil `joblib.dump`) sudah diletakkan di folder yang sama dengan file `app.py` ini.")
    st.info("Struktur folder yang benar:\n\n"
            "```text\n"
            "📂 Proyek-Anda/\n"
            "├── app.py\n"
            "└── cinematch_svd_model.pkl\n"
            "```")
else:
    # Mengambil daftar User ID dan Item ID asli langsung dari objek model
    try:
        all_raw_users = sorted([model.trainset.to_raw_uid(u) for u in model.trainset.all_users()])
        all_raw_items = sorted([model.trainset.to_raw_iid(i) for i in model.trainset.all_items()])
    except Exception:
        # Fallback jika model tidak menyimpan metadata internal trainset
        all_raw_users = None
        all_raw_items = None

    # --- KONTEN HALAMAN 1: BERANDA ---
    if menu == "🏠 Beranda":
        st.title("🎬 CineMatch: Sistem Rekomendasi Film Berbasis AI")
        st.subheader("Proyek Akhir Pelatihan PIJAK (Dicoding x IBM SkillsBuild)")
        
        st.markdown("""
        Aplikasi **CineMatch** dibangun menggunakan algoritma **Singular Value Decomposition (SVD)** yang dimuat via **Joblib**. 
        Sistem ini memprediksi ketertarikan (*rating prediction*) seorang pengguna terhadap film-film tertentu melalui pendekatan 
        *Collaborative Filtering*.
        
        Kelebihan sistem ini adalah kemampuannya mengenali pola tersembunyi (*latent factors*) antara preferensi user 
        dan karakteristik film, guna menghasilkan rekomendasi yang sangat personal dan akurat.
        """)
        
        st.markdown("---")
        st.subheader("📂 Integrasi Data Judul Film (Opsional)")
        uploaded_file = st.file_uploader("Jika Anda memiliki file `movies.csv` dari dataset MovieLens, unggah di sini agar ID Film dapat berubah menjadi Judul asli:", type=["csv"])
        if uploaded_file:
            st.session_state['movies_df'] = pd.read_csv(uploaded_file)
            st.success("✅ File `movies.csv` berhasil dimuat! Judul film akan ditampilkan secara otomatis di menu Rekomendasi.")

    # --- KONTEN HALAMAN 2: PREDIKSI RATING INDIVIDUAL ---
    elif menu == "🎯 Prediksi Rating":
        st.title("🎯 Prediksi Rating Individual")
        st.write("Gunakan fitur ini untuk memprediksi seberapa besar kecenderungan ketertarikan seorang user terhadap satu film spesifik (skala rating 1-5).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if all_raw_users:
                user_input = st.selectbox("Pilih User ID:", all_raw_users)
            else:
                user_input = st.text_input("Masukkan User ID (Contoh: 1):", value="1")
                
        with col2:
            if all_raw_items:
                item_input = st.selectbox("Pilih Movie ID:", all_raw_items)
            else:
                item_input = st.text_input("Masukkan Movie ID (Contoh: 10):", value="10")
        
        if st.button("Hitung Prediksi Skor"):
            try:
                # Konversi input otomatis ke integer jika berupa angka digital
                uid = int(user_input) if str(user_input).isdigit() else user_input
                iid = int(item_input) if str(item_input).isdigit() else item_input
                
                # Melakukan prediksi langsung ke objek SVD Joblib
                prediction = model.predict(uid, iid)
                score = prediction.est
                
                st.markdown("### 📊 Hasil Analisis Model SVD:")
                st.metric(label=f"Estimasi Rating dari User {uid} untuk Film {iid}", value=f"{score:.2f} / 5.00")
                
                if score >= 4.0:
                    st.success("🔥 **Rekomendasi Tinggi:** User ini diprediksi akan SANGAT menyukai film ini!")
                elif score >= 3.0:
                    st.info("👍 **Rekomendasi Sedang:** User diprediksi akan Cukup Menikmati film ini.")
                else:
                    st.warning("👎 **Rekomendasi Rendah:** Film ini kemungkinan kurang sesuai dengan selera User.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

    # --- KONTEN HALAMAN 3: REKOMENDASI PERSONAL TOP-10 ---
    elif menu == "🍿 Rekomendasi Personal":
        st.title("🍿 Rekomendasi Film Teratas (Top-10)")
        st.write("Pilih User ID untuk menyaring dan mengurutkan film terbaik yang belum pernah ditonton oleh user tersebut.")
        
        if all_raw_users and all_raw_items:
            target_user = st.selectbox("Pilih Target User ID:", all_raw_users)
            
            if st.button("Generate Rekomendasi"):
                with st.spinner("Model sedang menghitung matriks dekomposisi..."):
                    # Menghitung prediksi untuk semua item yang ada di database model
                    predictions = [model.predict(target_user, iid) for iid in all_raw_items]
                    
                    # Mengurutkan berdasarkan estimasi rating tertinggi
                    predictions.sort(key=lambda x: x.est, reverse=True)
                    top_10 = predictions[:10]
                    
                    # Memetakan hasil ke dataframe
                    recommendation_list = []
                    movies_df = st.session_state.get('movies_df', None)
                    
                    for idx, p in enumerate(top_10, 1):
                        item_id = p.iid
                        title = f"Movie ID {item_id}"
                        genres = "N/A (Unggah movies.csv di beranda untuk melihat)"
                        
                        if movies_df is not None:
                            match = movies_df[movies_df['movieId'] == item_id]
                            if not match.empty:
                                title = match['title'].values[0]
                                genres = match['genres'].values[0].replace('|', ', ')
                        
                        recommendation_list.append({
                            "No": idx,
                            "Movie ID": item_id,
                            "Judul Film": title,
                            "Genre": genres,
                            "Skor Prediksi SVD": f"{p.est:.2f}"
                        })
                    
                    df_res = pd.DataFrame(recommendation_list)
                    st.success(f"🎯 Berhasil menyusun 10 Film Terbaik untuk User ID {target_user}:")
                    st.dataframe(df_res.set_index("No"), use_container_width=True)
                    st.balloons()
        else:
            st.warning("⚠️ Fitur rekomendasi otomatis membutuhkan dataset internal yang terekstraksi. Silakan gunakan tab 'Prediksi Rating' untuk pengujian manual.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 | Proyek Capstone AI - Alumni PIJAK (Dicoding x IBM SkillsBuild)")