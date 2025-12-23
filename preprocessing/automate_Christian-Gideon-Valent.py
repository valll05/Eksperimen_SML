"""
Automate Preprocessing Script untuk Eksperimen SML
Author: Christian Gideon Valent
Dataset: Heart Disease Dataset (UCI)

Script ini mengotomasi proses preprocessing data untuk Machine Learning.
Tahapan yang dilakukan:
1. Memuat dataset dari file CSV atau download dari UCI
2. Memisahkan fitur dan target
3. Train-test split (80/20, stratified)
4. Feature scaling menggunakan StandardScaler
5. Menyimpan hasil preprocessing
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def create_raw_dataset(output_path):
    """
    Membuat dataset mentah dari UCI dan menyimpan ke file CSV.
    Heart Disease Dataset dari UCI ML Repository.
    
    Args:
        output_path: Path untuk menyimpan file CSV
        
    Returns:
        DataFrame: Dataset Heart Disease dalam format DataFrame
    """
    print("[INFO] Memuat dataset Heart Disease dari UCI...")
    
    # URL dataset Heart Disease dari UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names sesuai dokumentasi UCI
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        # Download dari UCI
        df = pd.read_csv(url, names=column_names, na_values='?')
        print("[OK] Dataset berhasil diunduh dari UCI Repository")
    except Exception as e:
        print(f"[WARNING] Gagal download dari UCI: {e}")
        print("[INFO] Menggunakan dataset alternatif...")
        # Alternatif: buat sample dataset jika download gagal
        df = create_sample_heart_dataset()
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease)
    # Original target: 0=no disease, 1-4=various stages of disease
    df['target'] = (df['target'] > 0).astype(int)
    
    # Simpan ke file CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset mentah disimpan ke: {output_path}")
    
    return df


def create_sample_heart_dataset():
    """
    Membuat sample Heart Disease dataset jika download gagal.
    Dataset ini mensimulasikan struktur Heart Disease UCI.
    """
    np.random.seed(42)
    n_samples = 303
    
    df = pd.DataFrame({
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 564, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(71, 202, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    return df


def load_data(filepath):
    """
    Memuat dataset dari file CSV.
    
    Args:
        filepath: Path ke file CSV
        
    Returns:
        DataFrame: Dataset yang dimuat
    """
    print(f"[INFO] Memuat data dari: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[OK] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def explore_data(df):
    """
    Melakukan eksplorasi dasar pada data.
    
    Args:
        df: DataFrame input
    """
    print("\n[EDA] Eksplorasi Data:")
    print("=" * 50)
    print(f"Jumlah Sampel: {df.shape[0]}")
    print(f"Jumlah Fitur: {df.shape[1] - 1}")  # Minus target
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplikat: {df.duplicated().sum()} baris")
    print(f"\nDistribusi Target:")
    print(f"   0 (No Disease): {(df['target'] == 0).sum()} sampel")
    print(f"   1 (Disease):    {(df['target'] == 1).sum()} sampel")


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Melakukan preprocessing data.
    
    Tahapan:
    1. Memisahkan fitur dan target
    2. Train-test split dengan stratified sampling
    3. Scaling fitur menggunakan StandardScaler
    
    Args:
        df: DataFrame input
        test_size: Proporsi data untuk testing (default: 0.2)
        random_state: Random seed untuk reproducibility
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    print("\n[PREPROCESSING] Memulai Preprocessing...")
    print("=" * 50)
    
    # 1. Memisahkan fitur dan target
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    X = df[feature_cols]
    y = df['target']
    
    print(f"[OK] Fitur: {feature_cols}")
    print(f"[OK] Target: target (0=No Disease, 1=Disease)")
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"\n[SPLIT] Split Data:")
    print(f"   Training set: {X_train.shape[0]} sampel ({(1-test_size)*100:.0f}%)")
    print(f"   Testing set: {X_test.shape[0]} sampel ({test_size*100:.0f}%)")
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    print(f"\n[OK] Feature Scaling selesai (StandardScaler)")
    print(f"   Mean setelah scaling (train): {X_train_scaled.mean().mean():.6f}")
    print(f"   Std setelah scaling (train): {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def save_results(X_train, X_test, y_train, y_test, scaler, output_dir):
    """
    Menyimpan hasil preprocessing ke file.
    
    Args:
        X_train: DataFrame fitur training yang sudah di-scale
        X_test: DataFrame fitur testing yang sudah di-scale
        y_train: Series target training
        y_test: Series target testing
        scaler: StandardScaler yang sudah di-fit
        output_dir: Direktori output
    """
    print(f"\n[SAVE] Menyimpan hasil ke: {output_dir}")
    print("=" * 50)
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabungkan fitur dan target untuk disimpan
    train_data = X_train.copy()
    train_data['target'] = y_train.values
    
    test_data = X_test.copy()
    test_data['target'] = y_test.values
    
    # Simpan ke file CSV
    train_path = os.path.join(output_dir, 'train_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    
    print(f"[OK] {train_path} ({train_data.shape[0]} sampel)")
    print(f"[OK] {test_path} ({test_data.shape[0]} sampel)")
    print(f"[OK] {scaler_path}")


def main():
    """
    Fungsi utama untuk menjalankan pipeline preprocessing.
    """
    print("=" * 60)
    print("AUTOMATE PREPROCESSING - EKSPERIMEN SML")
    print("Author: Christian Gideon Valent")
    print("Dataset: Heart Disease (UCI)")
    print("=" * 60)
    
    # Tentukan path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(base_dir, 'heart_raw', 'heart_raw.csv')
    output_dir = os.path.join(script_dir, 'heart_preprocessing')
    
    # 1. Buat/Muat dataset mentah
    if not os.path.exists(raw_data_path):
        df = create_raw_dataset(raw_data_path)
    else:
        df = load_data(raw_data_path)
    
    # 2. Eksplorasi data
    explore_data(df)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 4. Simpan hasil
    save_results(X_train, X_test, y_train, y_test, scaler, output_dir)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PREPROCESSING SELESAI!")
    print("=" * 60)
    
    # Ringkasan
    print("\n[SUMMARY] Ringkasan Preprocessing:")
    print(f"   - Dataset: Heart Disease ({df.shape[0]} sampel)")
    print(f"   - Fitur: {len(feature_names)} fitur")
    print(f"   - Missing Values: 0 (sudah di-handle)")
    print(f"   - Scaling: StandardScaler")
    print(f"   - Split: 80% train, 20% test (stratified)")
    print(f"   - Output: {output_dir}")


if __name__ == "__main__":
    main()
