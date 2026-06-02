# preprocessing/automate_membangun-mlsystem.py
import os
import zipfile
import urllib.request
import pandas as pd

def download_and_preprocess(raw_dir="namadataset_raw", output_dir="Membangun_model/namadataset_preprocessing"):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # URL Dataset MovieLens dari template capstone CineMatch
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(raw_dir, "ml-latest-small.zip")
    
    # Tahap 1: Data Loading & Download
    if not os.path.exists(zip_path):
        print("⬇️ Mengunduh dataset MovieLens Latest Small...")
        urllib.request.urlretrieve(url, zip_path)
        
    print("📦 Mengekstrak file zip dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    # Tahap 2: Load Data ke Pandas
    extracted_folder = os.path.join(raw_dir, "ml-latest-small")
    ratings = pd.read_csv(os.path.join(extracted_folder, "ratings.csv"))
    
    # Tahap 3: Preprocessing Otomatis (Cleaning Data)
    # Menghapus missing values jika ada & mengurutkan berdasarkan timestamp
    ratings.dropna(inplace=True)
    ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last', inplace=True)
    
    # Menyeleksi kolom utama yang dibutuhkan algoritma SVD Surprise
    processed_ratings = ratings[['userId', 'movieId', 'rating', 'timestamp']]
    
    # Tahap 4: Menyimpan Hasil Preprocessing yang Siap Dilatih
    output_path = os.path.join(output_dir, "cleaned_ratings.csv")
    processed_ratings.to_csv(output_path, index=False)
    print(f"✅ Preprocessing Selesai! Data bersih disimpan di: {output_path}")
    return output_path

if __name__ == "__main__":
    download_and_preprocess()
