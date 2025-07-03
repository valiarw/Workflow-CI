import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np # random tidak diperlukan jika hanya menggunakan np.random.seed
import os
import warnings
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse # Diperlukan untuk mengecek tipe tracking URI

# --- Konfigurasi Awal MLflow ---
mlflow.sklearn.autolog(disable=True) 

# Buat atau set eksperimen MLflow
mlflow.set_experiment("Heart-Failure Prediction") 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # --- Penanganan Path Data ---
    # Mendapatkan direktori skrip saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Membuat path ke file train.csv di dalam folder heart_preprocessing
    data_path = os.path.join(current_dir, "heart_preprocessing", "train.csv")

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found.")
        print("Please ensure 'train.csv' is in the 'heart_preprocessing' directory relative to your script, or provide the correct path.")
        sys.exit(1) 

    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        random_state=42,
        test_size=0.2,
        stratify=y
    )

    # Input example untuk logging model (ambil 5 baris pertama dari X_train)
    input_example = X_train.iloc[0:5] 

    # --- Ambil Hyperparameter dari Command Line (atau gunakan default) ---
    # sys.argv[0] adalah nama skrip itu sendiri
    # sys.argv[1] adalah argumen pertama (n_estimators)
    # sys.argv[2] adalah argumen kedua (max_depth)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Starting MLflow Run with n_estimators={n_estimators}, max_depth={max_depth}")

    # Memulai run MLflow untuk setiap kombinasi hyperparameter
    # Memberi nama run agar mudah diidentifikasi di UI MLflow
    with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}") as run:
        run_id = run.info.run_id
        
        # --- 1. Log Parameters ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state_model", 42) 
        mlflow.log_param("test_size_split", 0.2) #
        mlflow.log_param("stratify_split", "True") 

        # --- 2. Latih Model (Hanya Sekali) ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42, 
            n_jobs=-1 # 
        )
        model.fit(X_train, y_train)

        # --- 3. Evaluasi Model dan Log Metrik (Klasifikasi) ---
        predicted_disease = model.predict(X_test)
        predicted_proba = model.predict_proba(X_test)[:, 1] # Probabilitas untuk ROC AUC

        # Hitung metrik klasifikasi
        accuracy = accuracy_score(y_test, predicted_disease)
        precision = precision_score(y_test, predicted_disease, zero_division=0)
        recall = recall_score(y_test, predicted_disease, zero_division=0)
        f1 = f1_score(y_test, predicted_disease, zero_division=0)
        roc_auc = roc_auc_score(y_test, predicted_proba)

        # Log metrik ke MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        print(f"Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")

        # --- 4. Log Artefak (Plots) ---
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, predicted_disease)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) 
        plt.title(f'Confusion Matrix (n_est={n_estimators}, max_d={max_depth})') 
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_filepath = "confusion_matrix.png"
        plt.savefig(cm_filepath)
        mlflow.log_artifact(cm_filepath)
        plt.close() 

        # Log Classification Report sebagai file teks
        class_report_str = classification_report(y_test, predicted_disease, output_dict=False)
        report_filepath = "classification_report.txt"
        with open(report_filepath, "w") as f:
            f.write(class_report_str)
        mlflow.log_artifact(report_filepath)
        print("\nClassification Report:\n", class_report_str) 

        # --- 5. Log Model ke MLflow (Hanya sekali per run) ---
        # Infer signature untuk model
        signature = infer_signature(X_train, predicted_disease)
        
        # Dapatkan skema URI tracking untuk Model Registry
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Nama model yang akan terdaftar di MLflow Model Registry
        registered_model_name = "heart_disease_model_testing" 

        # Log model. Jika menggunakan remote store (non-file), coba daftarkan ke Model Registry
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model", 
                registered_model_name=registered_model_name,
                input_example=input_example,
                signature=signature
            )
            print(f"Model logged and registered as '{registered_model_name}' (Version {mlflow.active_run().info.run_id})")
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )
            print("Model logged (not registered - local file store does not support registry).")

    print(f"\nMLflow Run completed. View results at: {mlflow.get_tracking_uri()}")