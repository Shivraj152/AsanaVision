"""
Model execution and evaluation script for AsanaVision.
Executes training pipeline, outputs evaluation metrics, prints confusion matrix, and saves serialized model.
"""

import os
import argparse
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_model import ModelTrainer, INITIAL_3_POSES, SUPPORTED_POSES
from scripts.generate_synthetic_data import create_dataset

def main():
    parser = argparse.ArgumentParser(description="Train scikit-learn models for AsanaVision")
    parser.add_argument("--data", type=str, default="data/processed/pose_landmarks.csv")
    parser.add_argument("--initial_only", action="store_true", help="Train on initial 3 poses only (Tadasana, Bhujangasana, Vajrasana)")
    parser.add_argument("--auto_generate", action="store_true", default=True, help="Auto-generate dataset if data file is missing")

    args = parser.parse_args()

    # Check if dataset exists, if not generate it
    if not os.path.exists(args.data) and args.auto_generate:
        print(f"Dataset file '{args.data}' not found. Auto-generating pose landmark dataset...")
        target_poses = INITIAL_3_POSES if args.initial_only else SUPPORTED_POSES
        create_dataset(args.data, samples_per_pose=300, poses=target_poses)

    target_poses = INITIAL_3_POSES if args.initial_only else None

    print(f"\n========================================================")
    print(f"       ASANAVISION MODEL TRAINING & EVALUATION          ")
    print(f"========================================================\n")

    trainer = ModelTrainer(data_path=args.data)
    results = trainer.train_and_evaluate(poses_filter=target_poses)

    # Print Comparison Table
    print(f"{'Classifier':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for model_name, metrics in results.items():
        acc = f"{metrics['accuracy'] * 100:.2f}%"
        prec = f"{metrics['precision'] * 100:.2f}%"
        rec = f"{metrics['recall'] * 100:.2f}%"
        f1 = f"{metrics['f1_score'] * 100:.2f}%"
        print(f"{model_name:<22} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10}")

    print("-" * 75)

    # Print Random Forest Confusion Matrix
    rf_metrics = results["Random Forest"]
    labels = rf_metrics["labels"]
    cm = rf_metrics["confusion_matrix"]

    print("\nRandom Forest Confusion Matrix:")
    print(f"{'Class':<20} | " + " | ".join([f"{l[:6]:<6}" for l in labels]))
    print("-" * (23 + 9 * len(labels)))
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:<6}" for val in row])
        print(f"{labels[i]:<20} | {row_str}")

    # Save model and metrics JSON
    model_path, metrics_path = trainer.save_model()
    print(f"\nSaved primary classifier model to: {model_path}")
    print(f"Saved evaluation metrics JSON to: {metrics_path}\n")

if __name__ == "__main__":
    main()
