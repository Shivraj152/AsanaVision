"""
Raw image landmark extractor script for AsanaVision.
Extracts 3D pose landmarks and joint angles from raw yoga images in data/raw/<pose_name>/*.jpg.
"""

import os
import glob
import cv2
import argparse
import pandas as pd
import numpy as np
from typing import List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pose_detection import PoseDetector
from src.feature_extraction import extract_features_from_mediapipe, FEATURE_NAMES

def process_raw_dataset(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/pose_landmarks.csv"
):
    """
    Scans raw image directory structure data/raw/<pose_label>/*.jpg, extracts landmarks, and saves CSV.
    """
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory '{raw_dir}' does not exist.")
        print("Please place yoga pose images in subfolders matching pose names under data/raw/")
        print("Example: data/raw/Tadasana/img1.jpg")
        return

    detector = PoseDetector(static_image_mode=True, min_detection_confidence=0.5)
    records = []

    pose_folders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]

    if not pose_folders:
        print(f"No pose subdirectories found inside '{raw_dir}'.")
        return

    print(f"Found {len(pose_folders)} pose subfolders: {pose_folders}")

    total_processed = 0
    total_detected = 0

    for pose_label in pose_folders:
        folder_path = os.path.join(raw_dir, pose_label)
        image_paths = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))

        print(f"Processing '{pose_label}': {len(image_paths)} images found...")

        for img_path in image_paths:
            total_processed += 1
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            _, results = detector.process_frame(frame)
            feature_vector = extract_features_from_mediapipe(results)

            if feature_vector is not None:
                total_detected += 1
                row = list(feature_vector) + [pose_label]
                records.append(row)

    detector.close()

    if not records:
        print("No pose landmarks could be extracted from raw images.")
        return

    columns = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(records, columns=columns)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nLandmark Extraction Summary:")
    print(f"Total Images Processed: {total_processed}")
    print(f"Poses Detected & Extracted: {total_detected}")
    print(f"Extraction Efficiency: {round((total_detected / max(1, total_processed)) * 100, 2)}%")
    print(f"Saved dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from raw dataset images")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed/pose_landmarks.csv")
    args = parser.parse_args()

    process_raw_dataset(args.raw_dir, args.output)
