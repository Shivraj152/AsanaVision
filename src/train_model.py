"""
Model training, multi-algorithm comparison, evaluation, and serialization module for AsanaVision.
Trains Random Forest, SVM, KNN, and Logistic Regression on extracted landmark features across 20 yoga poses.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SUPPORTED_POSES = [
    "Anantasana",
    "Ardhakati Chakrasana",
    "Balasana",
    "Bhujangasana",
    "Halasana",
    "Kati Chakrasana",
    "Marjariasana",
    "Parvatasana",
    "Paschimottanasana",
    "Sarvangasana",
    "Setu Bandhasana",
    "Tadasana",
    "Trikonasana",
    "Ustrasana",
    "Utkatasana",
    "Vajrasana",
    "Viparita Karani",
    "Virabhadrasana I",
    "Virabhadrasana II",
    "Vrikshasana"
]

INITIAL_3_POSES = ["Tadasana", "Bhujangasana", "Vajrasana"]

class ModelTrainer:
    """Trains and compares multiple scikit-learn classifiers on landmark data."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=25, class_weight='balanced', random_state=42),
            "SVM": Pipeline([("scaler", StandardScaler()), ("svc", SVC(C=10.0, kernel='rbf', probability=True, random_state=42))]),
            "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5, weights='distance'))]),
            "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1500, random_state=42))])
        }
        self.evaluation_results = {}
        self.best_model_name = "Random Forest"
        self.best_model = None

    def load_and_clean_data(self, poses_filter: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads CSV dataset, cleans missing values, filters target poses if specified.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        
        # Clean missing values
        df = df.dropna()

        if poses_filter:
            df = df[df["label"].isin(poses_filter)]

        if df.empty:
            raise ValueError("Dataset is empty after loading and filtering.")

        X = df.drop(columns=["label"]).values
        y = df["label"].values

        return X, y

    def train_and_evaluate(
        self,
        poses_filter: list = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes train/test split, trains models, measures metrics, saves artifacts.
        """
        X, y = self.load_and_clean_data(poses_filter=poses_filter)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        labels = sorted(list(set(y)))
        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "labels": labels
            }

        self.evaluation_results = results
        self.best_model = self.models[self.best_model_name]

        return results

    def save_model(self, model_dir: str = "models") -> Tuple[str, str]:
        """
        Saves best model artifact and evaluation metrics to disk.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "yoga_model.pkl")
        metrics_path = os.path.join(model_dir, "model_metrics.json")

        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet.")

        # Save selected Random Forest model
        joblib.dump(self.best_model, model_path)

        # Save metrics to JSON
        with open(metrics_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

        return model_path, metrics_path
