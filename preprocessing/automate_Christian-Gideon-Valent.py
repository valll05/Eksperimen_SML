"""
Automate Preprocessing Script untuk Eksperimen SML
Author: Christian Gideon Valent
Dataset: Iris Dataset

Script ini mengotomasi proses preprocessing data untuk Machine Learning.
Tahapan yang dilakukan:
1. Memuat dataset dari file CSV
2. Memisahkan fitur dan target
3. Train-test split (80/20, stratified)
4. Feature scaling menggunakan StandardScaler
5. Menyimpan hasil preprocessing
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
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
    Membuat dataset mentah dari sklearn dan menyimpan ke file CSV.
    
    Args:
        output_path: Path untuk menyimpan file CSV
        
    Returns:
        DataFrame: Dataset Iris dalam format DataFrame
    """
    print("[INFO] Memuat dataset Iris dari sklearn...")
    iris = load_iris()
    
    # Konversi ke DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Simpan ke file CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset mentah disimpan ke: {output_path}")
    
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
    print(f"Jumlah Fitur: {df.shape[1] - 2}")  # Minus target dan species
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplikat: {df.duplicated().sum()} baris")
    print(f"\nDistribusi Target:\n{df['species'].value_counts()}")


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
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    X = df[feature_cols]
    y = df['target']
    
    print(f"[OK] Fitur: {feature_cols}")
    print(f"[OK] Target: target (0=setosa, 1=versicolor, 2=virginica)")
    
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
    print("Dataset: Iris")
    print("=" * 60)
    
    # Tentukan path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    raw_data_path = os.path.join(base_dir, 'iris_raw', 'iris_raw.csv')
    output_dir = os.path.join(script_dir, 'iris_preprocessing')
    
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
    print(f"   - Dataset: Iris ({df.shape[0]} sampel)")
    print(f"   - Fitur: {len(feature_names)} fitur numerik")
    print(f"   - Missing Values: 0")
    print(f"   - Scaling: StandardScaler")
    print(f"   - Split: 80% train, 20% test (stratified)")
    print(f"   - Output: {output_dir}")


if __name__ == "__main__":
    main()
