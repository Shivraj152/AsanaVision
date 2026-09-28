"""
Unit tests for AsanaVision feature extraction, angle math, posture feedback across 20 poses, and predictor.
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extraction import (
    calculate_angle_3d, calculate_2d_angle, normalize_landmarks, extract_joint_angles
)
from src.posture_feedback import PostureEvaluator, POSE_TARGET_RULES
from src.predict import PosePredictor
from src.reference_guides import YOGA_POSE_GUIDES

class TestPoseModule(unittest.TestCase):

    def test_twenty_poses_guides_present(self):
        self.assertEqual(len(YOGA_POSE_GUIDES), 20)
        for pose, info in YOGA_POSE_GUIDES.items():
            self.assertIn("english_name", info)
            self.assertIn("category", info)
            self.assertIn("image_file", info)
            self.assertIn("instructions", info)
            self.assertIn("target_angles_summary", info)
            self.assertIn("sanskrit_script", info)
            self.assertIn("difficulty", info)

    def test_twenty_poses_rules_present(self):
        self.assertEqual(len(POSE_TARGET_RULES), 20)

    def test_angle_3d_perpendicular(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([1.0, 1.0, 0.0])
        angle = calculate_angle_3d(p1, p2, p3)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_angle_3d_straight(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        p3 = np.array([0.0, 2.0, 0.0])
        angle = calculate_angle_3d(p1, p2, p3)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_normalize_landmarks(self):
        dummy_lms = np.random.uniform(-10.0, 10.0, size=(33, 3))
        norm_lms = normalize_landmarks(dummy_lms)
        self.assertEqual(norm_lms.shape, (33, 3))

    def test_posture_evaluator_vrikshasana(self):
        evaluator = PostureEvaluator()
        angles = {
            "left_knee_angle": 170.0,
            "right_knee_angle": 60.0,
            "torso_vertical_angle": 5.0
        }
        res = evaluator.evaluate("Vrikshasana", angles, 0.95)
        self.assertTrue(res["is_correct"])
        self.assertEqual(res["status"], "Correct Posture")

    def test_posture_evaluator_virabhadrasana_ii(self):
        evaluator = PostureEvaluator()
        angles = {
            "right_knee_angle": 90.0,
            "left_knee_angle": 170.0,
            "left_shoulder_angle": 90.0,
            "right_shoulder_angle": 90.0
        }
        res = evaluator.evaluate("Virabhadrasana II", angles, 0.95)
        self.assertTrue(res["is_correct"])

    def test_predictor_missing_model(self):
        predictor = PosePredictor(model_path="non_existent_model.pkl")
        label, conf = predictor.predict(np.zeros(109))
        self.assertEqual(label, "Unknown (Model Not Trained)")
        self.assertEqual(conf, 0.0)

if __name__ == "__main__":
    unittest.main()
