"""
Modelling Tuning Script untuk Eksperimen SML - Kriteria Skilled/Advance
Author: Christian Gideon Valent

Script ini melatih model ML dengan hyperparameter tuning dan MLflow manual logging.

Kriteria Skilled:
- Hyperparameter tuning dengan GridSearchCV
- Manual logging metriks (sama dengan autolog)
- Logging model artifacts

Kriteria Advance:
- DagsHub integration untuk online tracking
- Minimal 2 artefak tambahan:
  1. Confusion Matrix Plot
  2. Feature Importance Plot  
  3. Classification Report (text file)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI MLFLOW TRACKING
# ============================================================
# Untuk lokal, gunakan file-based tracking (tidak perlu server)
script_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_dir = os.path.join(script_dir, 'mlruns')
mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

# UNTUK DAGSHUB (Kriteria Advance) - Uncomment bagian ini:
# import dagshub
# dagshub.init(repo_owner='valll05', repo_name='Eksperimen_SML', mlflow=True)

# Buat experiment
mlflow.set_experiment("Iris_Classification_Tuning")


def load_preprocessed_data():
    """Load data yang sudah dipreprocessing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'iris_preprocessing')
    
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test, feature_cols


def save_confusion_matrix(y_true, y_pred, output_path):
    """Simpan confusion matrix sebagai gambar."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def save_feature_importance(model, feature_names, output_path):
    """Simpan feature importance sebagai gambar."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn(importance[indices] / max(importance))
    plt.barh(range(len(indices)), importance[indices], color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def save_classification_report(y_true, y_pred, output_path):
    """Simpan classification report ke file text."""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Setosa', 'Versicolor', 'Virginica'])
    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    return output_path


def train_with_hyperparameter_tuning():
    """
    Melatih model dengan hyperparameter tuning dan manual logging.
    Memenuhi kriteria Skilled dan Advance.
    """
    print("=" * 60)
    print("MODELLING TUNING - KRITERIA SKILLED/ADVANCE")
    print("Hyperparameter Tuning + Manual Logging")
    print("=" * 60)
    
    # Load data
    print("\n[INFO] Loading preprocessed data...")
    X_train, X_test, y_train, y_test, feature_cols = load_preprocessed_data()
    print(f"[OK] Training data: {X_train.shape[0]} samples")
    print(f"[OK] Testing data: {X_test.shape[0]} samples")
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    print("\n[INFO] Performing GridSearchCV...")
    print(f"Parameter grid: {param_grid}")
    
    # GridSearchCV
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"\n[OK] Best parameters: {best_params}")
    print(f"[OK] Best CV score: {best_score:.4f}")
    
    # Train final model with best params
    print("\n[INFO] Training final model with best parameters...")
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n[RESULT] Test Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Create temp directory for artifacts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, 'mlflow_artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # ============================================================
    # MLFLOW MANUAL LOGGING (Kriteria Skilled)
    # ============================================================
    print("\n[INFO] Logging to MLflow (Manual Logging)...")
    
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        # Log parameters (manual)
        mlflow.log_param("n_estimators", best_params['n_estimators'])
        mlflow.log_param("max_depth", best_params['max_depth'])
        mlflow.log_param("min_samples_split", best_params['min_samples_split'])
        mlflow.log_param("min_samples_leaf", best_params['min_samples_leaf'])
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        
        # Log metrics (manual - sama dengan yang di-log oleh autolog)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_best_score", best_score)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # ============================================================
        # ARTEFAK TAMBAHAN (Kriteria Advance - minimal 2)
        # ============================================================
        print("\n[INFO] Creating additional artifacts (Advance criteria)...")
        
        # Artefak 1: Confusion Matrix Plot
        cm_path = os.path.join(artifacts_dir, 'training_confusion_matrix.png')
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        print(f"   [OK] Logged: training_confusion_matrix.png")
        
        # Artefak 2: Feature Importance Plot
        fi_path = os.path.join(artifacts_dir, 'feature_importance.png')
        save_feature_importance(best_model, feature_cols, fi_path)
        mlflow.log_artifact(fi_path)
        print(f"   [OK] Logged: feature_importance.png")
        
        # Artefak 3: Classification Report (text)
        cr_path = os.path.join(artifacts_dir, 'classification_report.txt')
        save_classification_report(y_test, y_pred, cr_path)
        mlflow.log_artifact(cr_path)
        print(f"   [OK] Logged: classification_report.txt")
        
        # Artefak 4: Metric Info JSON
        metric_info = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "cv_best_score": float(best_score),
            "best_params": best_params,
            "n_samples_train": int(X_train.shape[0]),
            "n_samples_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1])
        }
        mi_path = os.path.join(artifacts_dir, 'metric_info.json')
        with open(mi_path, 'w') as f:
            json.dump(metric_info, f, indent=2)
        mlflow.log_artifact(mi_path)
        print(f"   [OK] Logged: metric_info.json")
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] Run ID: {run_id}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] TRAINING DENGAN TUNING SELESAI!")
    print("=" * 60)
    print("\n[INFO] Untuk melihat hasil:")
    print("   1. Jalankan: mlflow ui")
    print("   2. Buka: http://127.0.0.1:5000")
    print("   3. Pilih experiment 'Iris Classification Tuning'")
    
    return best_model


if __name__ == "__main__":
    train_with_hyperparameter_tuning()
