"""
Landmark feature extraction and pose normalization for AsanaVision.
Extracts scale-invariant and position-invariant 3D keypoint features and joint angles.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# MediaPipe Pose Landmark Index Constants
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

FEATURE_NAMES = [
    f"{axis}_{i}" for i in range(33) for axis in ["norm_x", "norm_y", "norm_z"]
] + [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "torso_vertical_angle",
    "shoulder_horizontal_angle"
]

def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculates 3D interior angle at vertex p2 formed by p1-p2-p3 in degrees [0, 180].
    
    :param p1: First endpoint (x, y, z)
    :param p2: Vertex point (x, y, z)
    :param p3: Second endpoint (x, y, z)
    :return: Angle in degrees
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))

def calculate_2d_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Calculates 2D planar angle at vertex p2 formed by p1-p2-p3 in degrees [0, 180].
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cosine_angle)))

def extract_joint_angles(landmarks_3d: np.ndarray) -> Dict[str, float]:
    """
    Extracts key biomechanical joint angles from 33 3D landmark array (33, 3).
    
    :param landmarks_3d: Array of shape (33, 3) representing (x, y, z) coordinates.
    :return: Dictionary of named angle features in degrees.
    """
    # Key landmark coordinates
    l_sh = landmarks_3d[LEFT_SHOULDER]
    r_sh = landmarks_3d[RIGHT_SHOULDER]
    l_el = landmarks_3d[LEFT_ELBOW]
    r_el = landmarks_3d[RIGHT_ELBOW]
    l_wr = landmarks_3d[LEFT_WRIST]
    r_wr = landmarks_3d[RIGHT_WRIST]
    l_hip = landmarks_3d[LEFT_HIP]
    r_hip = landmarks_3d[RIGHT_HIP]
    l_knee = landmarks_3d[LEFT_KNEE]
    r_knee = landmarks_3d[RIGHT_KNEE]
    l_ank = landmarks_3d[LEFT_ANKLE]
    r_ank = landmarks_3d[RIGHT_ANKLE]
    
    # Calculate key angles
    angles = {
        "left_elbow_angle": calculate_angle_3d(l_sh, l_el, l_wr),
        "right_elbow_angle": calculate_angle_3d(r_sh, r_el, r_wr),
        "left_knee_angle": calculate_angle_3d(l_hip, l_knee, l_ank),
        "right_knee_angle": calculate_angle_3d(r_hip, r_knee, r_ank),
        "left_shoulder_angle": calculate_angle_3d(l_hip, l_sh, l_el),
        "right_shoulder_angle": calculate_angle_3d(r_hip, r_sh, r_el),
        "left_hip_angle": calculate_angle_3d(l_sh, l_hip, l_knee),
        "right_hip_angle": calculate_angle_3d(r_sh, r_hip, r_knee),
    }
    
    # Torso alignment relative to vertical (y-axis)
    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0
    torso_vector = shoulder_mid - hip_mid
    vertical_vector = np.array([0.0, -1.0, 0.0]) # MediaPipe y increases downwards
    
    torso_norm = np.linalg.norm(torso_vector)
    if torso_norm > 1e-6:
        cos_torso = np.dot(torso_vector, vertical_vector) / torso_norm
        angles["torso_vertical_angle"] = float(np.degrees(np.arccos(np.clip(cos_torso, -1.0, 1.0))))
    else:
        angles["torso_vertical_angle"] = 0.0
        
    # Shoulder tilt angle (horizontal alignment)
    shoulder_vector = r_sh - l_sh
    horizontal_vector = np.array([1.0, 0.0, 0.0])
    sh_norm = np.linalg.norm(shoulder_vector)
    if sh_norm > 1e-6:
        cos_sh = np.dot(shoulder_vector, horizontal_vector) / sh_norm
        angles["shoulder_horizontal_angle"] = float(np.degrees(np.arccos(np.clip(cos_sh, -1.0, 1.0))))
    else:
        angles["shoulder_horizontal_angle"] = 0.0
        
    return angles

def normalize_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Translates landmarks relative to hip center and scales by torso length.
    
    :param landmarks_3d: Array of shape (33, 3) representing (x, y, z) coordinates.
    :return: Normalized array of shape (33, 3).
    """
    l_sh = landmarks_3d[LEFT_SHOULDER]
    r_sh = landmarks_3d[RIGHT_SHOULDER]
    l_hip = landmarks_3d[LEFT_HIP]
    r_hip = landmarks_3d[RIGHT_HIP]
    
    hip_center = (l_hip + r_hip) / 2.0
    translated = landmarks_3d - hip_center
    
    shoulder_center = (l_sh + r_sh) / 2.0
    torso_size = np.linalg.norm(shoulder_center - hip_center)
    
    if torso_size < 1e-6:
        torso_size = 1.0
        
    normalized = translated / torso_size
    return normalized

def extract_features_from_mediapipe(results: Any) -> Optional[np.ndarray]:
    """
    Processes MediaPipe pose landmarks output and returns a flat 1D feature vector.
    Feature vector structure: 99 normalized landmark coordinates (x, y, z * 33) + 10 joint angles = 109 floats.
    
    :param results: MediaPipe pose processing result object.
    :return: 1D numpy array of 109 feature values, or None if no pose detected.
    """
    if results is None or not results.pose_landmarks:
        return None
        
    raw_coords = []
    for lm in results.pose_landmarks.landmark:
        raw_coords.append([lm.x, lm.y, lm.z])
        
    landmarks_3d = np.array(raw_coords, dtype=np.float32)
    
    # Normalize position and scale
    norm_lms = normalize_landmarks(landmarks_3d)
    
    # Extract joint angles
    angles_dict = extract_joint_angles(landmarks_3d)
    
    # Flatten normalized coordinates (99 floats)
    flat_norm = norm_lms.flatten()
    
    # Order angle features deterministically
    angle_vals = [
        angles_dict["left_elbow_angle"],
        angles_dict["right_elbow_angle"],
        angles_dict["left_knee_angle"],
        angles_dict["right_knee_angle"],
        angles_dict["left_shoulder_angle"],
        angles_dict["right_shoulder_angle"],
        angles_dict["left_hip_angle"],
        angles_dict["right_hip_angle"],
        angles_dict["torso_vertical_angle"],
        angles_dict["shoulder_horizontal_angle"]
    ]
    
    feature_vector = np.concatenate([flat_norm, angle_vals])
    return feature_vector
