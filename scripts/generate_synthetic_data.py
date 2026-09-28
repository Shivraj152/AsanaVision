"""
Dataset landmark generator for AsanaVision.
Generates biomechanically grounded landmark features for bootstrapping model training across 20 yoga poses.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extraction import FEATURE_NAMES, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP

ALL_POSES = [
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

# Canonical baseline joint angle configurations per pose (10 angles in degrees)
# Order: [left_elbow, right_elbow, left_knee, right_knee, left_shoulder, right_shoulder, left_hip, right_hip, torso_vertical, shoulder_horizontal]
POSE_ANGLE_PROFILES: Dict[str, List[float]] = {
    "Tadasana": [175.0, 175.0, 175.0, 175.0, 15.0, 15.0, 170.0, 170.0, 5.0, 2.0],
    "Bhujangasana": [140.0, 140.0, 165.0, 165.0, 45.0, 45.0, 150.0, 150.0, 50.0, 2.0],
    "Vajrasana": [170.0, 170.0, 30.0, 30.0, 20.0, 20.0, 90.0, 90.0, 5.0, 2.0],
    "Anantasana": [160.0, 160.0, 160.0, 160.0, 90.0, 30.0, 140.0, 170.0, 80.0, 15.0],
    "Ardhakati Chakrasana": [170.0, 170.0, 170.0, 170.0, 165.0, 20.0, 165.0, 165.0, 30.0, 25.0],
    "Kati Chakrasana": [150.0, 150.0, 170.0, 170.0, 85.0, 85.0, 170.0, 170.0, 5.0, 40.0],
    "Marjariasana": [170.0, 170.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 85.0, 2.0],
    "Parvatasana": [170.0, 170.0, 165.0, 165.0, 160.0, 160.0, 80.0, 80.0, 55.0, 2.0],
    "Sarvangasana": [175.0, 175.0, 175.0, 175.0, 15.0, 15.0, 175.0, 175.0, 165.0, 2.0],
    "Viparita Karani": [170.0, 170.0, 170.0, 170.0, 20.0, 20.0, 90.0, 90.0, 85.0, 2.0],
    "Vrikshasana": [160.0, 160.0, 170.0, 60.0, 160.0, 160.0, 170.0, 130.0, 5.0, 2.0],
    "Trikonasana": [170.0, 170.0, 170.0, 170.0, 165.0, 165.0, 130.0, 130.0, 50.0, 35.0],
    "Virabhadrasana I": [170.0, 170.0, 165.0, 95.0, 165.0, 165.0, 160.0, 100.0, 10.0, 2.0],
    "Virabhadrasana II": [175.0, 175.0, 165.0, 90.0, 90.0, 90.0, 165.0, 100.0, 5.0, 5.0],
    "Utkatasana": [170.0, 170.0, 100.0, 100.0, 160.0, 160.0, 110.0, 110.0, 35.0, 2.0],
    "Setu Bandhasana": [170.0, 170.0, 90.0, 90.0, 30.0, 30.0, 155.0, 155.0, 75.0, 2.0],
    "Balasana": [165.0, 165.0, 30.0, 30.0, 160.0, 160.0, 30.0, 30.0, 85.0, 2.0],
    "Paschimottanasana": [160.0, 160.0, 170.0, 170.0, 70.0, 70.0, 40.0, 40.0, 80.0, 2.0],
    "Ustrasana": [165.0, 165.0, 90.0, 90.0, 140.0, 140.0, 150.0, 150.0, 55.0, 2.0],
    "Halasana": [170.0, 170.0, 170.0, 170.0, 25.0, 25.0, 60.0, 60.0, 150.0, 2.0]
}

def generate_pose_sample(pose_name: str, noise_std: float = 2.5) -> np.ndarray:
    """
    Generates a single feature vector (99 normalized coordinates + 10 angles = 109 values).
    """
    profile = POSE_ANGLE_PROFILES.get(pose_name, POSE_ANGLE_PROFILES["Tadasana"])
    angles = np.array(profile) + np.random.normal(0, noise_std, size=10)
    angles = np.clip(angles, 0.0, 180.0)

    coords = np.zeros((33, 3), dtype=np.float32)

    # Base structural keypoint templates per pose category
    if pose_name in ["Tadasana", "Ardhakati Chakrasana", "Kati Chakrasana", "Vrikshasana"]:
        coords[:, 1] = np.linspace(-1.0, 1.0, 33)
    elif pose_name in ["Virabhadrasana I", "Virabhadrasana II", "Utkatasana", "Trikonasana"]:
        coords[:, 0] = np.linspace(-0.8, 0.8, 33)
        coords[:, 1] = np.linspace(-0.6, 0.8, 33)
    elif pose_name in ["Bhujangasana", "Marjariasana", "Ustrasana", "Balasana"]:
        coords[:, 0] = np.linspace(-1.0, 1.0, 33)
    elif pose_name in ["Sarvangasana", "Viparita Karani", "Parvatasana", "Halasana"]:
        coords[:, 1] = np.linspace(1.0, -1.0, 33)
    elif pose_name in ["Vajrasana", "Paschimottanasana", "Setu Bandhasana", "Anantasana"]:
        coords[:, 1] = np.linspace(-0.5, 0.5, 33)
        coords[:, 0] = np.linspace(-0.7, 0.7, 33)
    else:
        coords = np.random.uniform(-0.8, 0.8, size=(33, 3))

    coords += np.random.normal(0, 0.03, size=(33, 3))
    flat_coords = coords.flatten()

    return np.concatenate([flat_coords, angles])

def create_dataset(
    output_path: str,
    samples_per_pose: int = 350,
    poses: List[str] = ALL_POSES
) -> pd.DataFrame:
    """
    Creates and saves CSV dataset with features and pose labels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data_rows = []
    for pose in poses:
        for _ in range(samples_per_pose):
            features = generate_pose_sample(pose)
            row = list(features) + [pose]
            data_rows.append(row)

    columns = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(data_rows, columns=columns)

    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} samples across {len(poses)} poses at: {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate landmark dataset for AsanaVision")
    parser.add_argument("--output", type=str, default="data/processed/pose_landmarks.csv")
    parser.add_argument("--samples", type=int, default=350)
    parser.add_argument("--initial_only", action="store_true", help="Generate only 3 initial prototype poses")
    
    args = parser.parse_args()
    target_poses = INITIAL_3_POSES if args.initial_only else ALL_POSES
    create_dataset(args.output, samples_per_pose=args.samples, poses=target_poses)
