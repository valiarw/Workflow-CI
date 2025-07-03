import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

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

    input_example = X_train[0:5]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Starting MLflow Run with n_estimators={n_estimators}, max_depth={max_depth}")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )
        model.fit(X_train, y_train)
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)