"""
Modelling Script untuk Eksperimen SML - Kriteria Basic
Author: Christian Gideon Valent

Script ini melatih model Machine Learning dengan MLflow Tracking.
Menggunakan autolog dari MLflow untuk logging otomatis.

Kriteria Basic:
- Melatih model ML (Scikit-Learn) dengan MLflow Tracking UI lokal
- Menggunakan autolog dari MLflow
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings('ignore')

# Set MLflow tracking URI ke lokal (file-based)
script_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_dir = os.path.join(script_dir, 'mlruns')
mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

# Buat experiment baru
mlflow.set_experiment("Iris_Classification")


def load_preprocessed_data():
    """Load data yang sudah dipreprocessing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'iris_preprocessing')
    
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # Pisahkan fitur dan target
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test


def train_model_with_autolog():
    """
    Melatih model dengan MLflow autolog.
    Kriteria Basic: Menggunakan autolog untuk logging otomatis.
    """
    print("=" * 60)
    print("MODELLING - KRITERIA BASIC")
    print("MLflow Autolog")
    print("=" * 60)
    
    # Load data
    print("\n[INFO] Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"[OK] Training data: {X_train.shape[0]} samples")
    print(f"[OK] Testing data: {X_test.shape[0]} samples")
    
    # Enable autolog
    print("\n[INFO] Enabling MLflow autolog...")
    mlflow.sklearn.autolog()
    
    # Train model
    print("\n[INFO] Training RandomForestClassifier...")
    with mlflow.start_run(run_name="Basic_Autolog"):
        # Model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[RESULT] Accuracy: {accuracy:.4f}")
        print(f"\n[INFO] Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\n[OK] Model logged to MLflow!")
        print("[INFO] Run 'mlflow ui' to view the dashboard")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] TRAINING SELESAI!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train_model_with_autolog()
