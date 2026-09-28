"""
Biomechanics and Posture Feedback Engine for AsanaVision.
Provides rule-based evaluation of posture correctness, accuracy percentage calculation, and real-time actionable feedback across 20 yoga poses.
"""

from typing import Dict, List, Tuple, Any
import numpy as np

# Target angle definitions per pose (20 Poses)
POSE_TARGET_RULES = {
    "Tadasana": {
        "left_knee_angle": (155.0, 180.0, "Straighten your left knee."),
        "right_knee_angle": (155.0, 180.0, "Straighten your right knee."),
        "torso_vertical_angle": (0.0, 18.0, "Keep your spine upright and vertical."),
        "shoulder_horizontal_angle": (0.0, 15.0, "Align your shoulders horizontally.")
    },
    "Bhujangasana": {
        "left_elbow_angle": (120.0, 180.0, "Extend your left arm to support your chest."),
        "right_elbow_angle": (120.0, 180.0, "Extend your right arm to support your chest."),
        "torso_vertical_angle": (25.0, 75.0, "Arch your back and lift your chest upwards."),
        "left_knee_angle": (150.0, 180.0, "Keep your left leg fully extended behind you."),
        "right_knee_angle": (150.0, 180.0, "Keep your right leg fully extended behind you.")
    },
    "Vajrasana": {
        "left_knee_angle": (0.0, 50.0, "Fold your left knee completely under your hips."),
        "right_knee_angle": (0.0, 50.0, "Fold your right knee completely under your hips."),
        "torso_vertical_angle": (0.0, 20.0, "Straighten your spine while seated on your heels."),
        "shoulder_horizontal_angle": (0.0, 15.0, "Relax and level your shoulders.")
    },
    "Anantasana": {
        "left_knee_angle": (140.0, 180.0, "Keep your raised leg straight."),
        "right_knee_angle": (140.0, 180.0, "Keep your resting leg fully extended."),
        "left_hip_angle": (100.0, 180.0, "Lift your top leg higher towards vertical.")
    },
    "Ardhakati Chakrasana": {
        "left_shoulder_angle": (140.0, 180.0, "Extend your raised arm straight overhead."),
        "torso_vertical_angle": (15.0, 50.0, "Bend laterally from your waist to the side."),
        "left_knee_angle": (150.0, 180.0, "Keep both knees straight without bending.")
    },
    "Kati Chakrasana": {
        "shoulder_horizontal_angle": (10.0, 60.0, "Rotate your upper torso and shoulders into the twist."),
        "left_knee_angle": (150.0, 180.0, "Keep your hips stable and legs straight."),
        "right_knee_angle": (150.0, 180.0, "Keep your hips stable and legs straight.")
    },
    "Marjariasana": {
        "left_elbow_angle": (145.0, 180.0, "Keep arms straight under your shoulders."),
        "right_elbow_angle": (145.0, 180.0, "Keep arms straight under your shoulders."),
        "left_knee_angle": (75.0, 110.0, "Position knees at a 90 degree angle under hips.")
    },
    "Parvatasana": {
        "left_elbow_angle": (150.0, 180.0, "Press palms firmly and extend elbows straight."),
        "right_elbow_angle": (150.0, 180.0, "Press palms firmly and extend elbows straight."),
        "left_hip_angle": (60.0, 120.0, "Push your hips up and back to form an inverted V.")
    },
    "Sarvangasana": {
        "torso_vertical_angle": (140.0, 180.0, "Lift your torso vertically upright onto shoulders."),
        "left_knee_angle": (150.0, 180.0, "Extend legs straight up towards the ceiling."),
        "right_knee_angle": (150.0, 180.0, "Extend legs straight up towards the ceiling.")
    },
    "Viparita Karani": {
        "left_knee_angle": (150.0, 180.0, "Extend knees straight resting vertically."),
        "right_knee_angle": (150.0, 180.0, "Extend knees straight resting vertically."),
        "left_hip_angle": (70.0, 115.0, "Maintain 90 degree angle between legs and torso.")
    },
    "Vrikshasana": {
        "left_knee_angle": (155.0, 180.0, "Straighten your standing left leg."),
        "right_knee_angle": (35.0, 90.0, "Fold right knee and rest sole on inner thigh."),
        "torso_vertical_angle": (0.0, 15.0, "Keep your torso upright and spine straight.")
    },
    "Trikonasana": {
        "left_knee_angle": (150.0, 180.0, "Keep your left knee fully extended."),
        "right_knee_angle": (150.0, 180.0, "Keep your right knee fully extended."),
        "torso_vertical_angle": (35.0, 70.0, "Tilt torso laterally over your lead leg."),
        "left_shoulder_angle": (150.0, 180.0, "Extend both arms wide in a straight T-shape.")
    },
    "Virabhadrasana I": {
        "right_knee_angle": (80.0, 110.0, "Bend front right knee to a 90 degree angle."),
        "left_knee_angle": (150.0, 180.0, "Extend back left leg straight with heel grounded."),
        "left_shoulder_angle": (140.0, 180.0, "Reach both arms straight overhead."),
        "right_shoulder_angle": (140.0, 180.0, "Reach both arms straight overhead.")
    },
    "Virabhadrasana II": {
        "right_knee_angle": (75.0, 105.0, "Bend front right knee to 90 degrees over ankle."),
        "left_knee_angle": (150.0, 180.0, "Keep back left leg straight and strong."),
        "left_shoulder_angle": (75.0, 105.0, "Extend rear arm parallel to floor."),
        "right_shoulder_angle": (75.0, 105.0, "Extend front arm parallel to floor.")
    },
    "Utkatasana": {
        "left_knee_angle": (80.0, 130.0, "Bend knees deeper into the chair squat."),
        "right_knee_angle": (80.0, 130.0, "Bend knees deeper into the chair squat."),
        "torso_vertical_angle": (20.0, 50.0, "Keep chest open and torso angled forward."),
        "left_shoulder_angle": (140.0, 180.0, "Extend arms overhead alongside your ears.")
    },
    "Setu Bandhasana": {
        "left_hip_angle": (130.0, 175.0, "Press feet into floor and lift hips higher."),
        "right_hip_angle": (130.0, 175.0, "Press feet into floor and lift hips higher."),
        "left_knee_angle": (70.0, 110.0, "Keep knees bent with feet flat on mat."),
        "right_knee_angle": (70.0, 110.0, "Keep knees bent with feet flat on mat.")
    },
    "Balasana": {
        "left_knee_angle": (0.0, 45.0, "Fold knees completely under hips."),
        "right_knee_angle": (0.0, 45.0, "Fold knees completely under hips."),
        "left_hip_angle": (0.0, 45.0, "Rest chest comfortably on your thighs."),
        "right_hip_angle": (0.0, 45.0, "Rest chest comfortably on your thighs.")
    },
    "Paschimottanasana": {
        "left_knee_angle": (150.0, 180.0, "Keep your left leg fully extended straight."),
        "right_knee_angle": (150.0, 180.0, "Keep your right leg fully extended straight."),
        "left_hip_angle": (20.0, 60.0, "Hinge forward from hips over thighs."),
        "right_hip_angle": (20.0, 60.0, "Hinge forward from hips over thighs.")
    },
    "Ustrasana": {
        "left_knee_angle": (75.0, 110.0, "Kneel upright with thighs vertical."),
        "right_knee_angle": (75.0, 110.0, "Kneel upright with thighs vertical."),
        "torso_vertical_angle": (30.0, 80.0, "Arch upper spine back towards your heels."),
        "left_shoulder_angle": (120.0, 180.0, "Reach hands down to grasp your heels.")
    },
    "Halasana": {
        "left_knee_angle": (145.0, 180.0, "Keep legs fully straight behind your head."),
        "right_knee_angle": (145.0, 180.0, "Keep legs fully straight behind your head."),
        "left_hip_angle": (40.0, 90.0, "Lift hips high into inverted plow fold.")
    }
}

class PostureEvaluator:
    """Evaluates posture correctness and calculates biomechanical accuracy score."""

    def __init__(self, rules: Dict[str, Dict] = None):
        self.rules = rules if rules is not None else POSE_TARGET_RULES

    def evaluate(
        self,
        pose_name: str,
        angles: Dict[str, float],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Evaluates detected pose against target biomechanical joint rules.

        :param pose_name: Name of detected yoga pose.
        :param angles: Dict of computed joint angles.
        :param confidence: Classification model confidence (0.0 to 1.0).
        :return: Dict containing correctness status, accuracy percentage, feedback list, and angle comparisons.
        """
        if pose_name not in self.rules:
            return {
                "is_correct": True,
                "accuracy_score": 100.0,
                "status": "Detected",
                "feedback": ["Pose detected. Maintain your posture."],
                "angle_analysis": {}
            }

        pose_rules = self.rules[pose_name]
        feedback_messages = []
        angle_analysis = {}
        joint_scores = []
        all_passed = True

        for angle_name, (min_val, max_val, suggestion) in pose_rules.items():
            current_val = angles.get(angle_name, 0.0)
            passed = min_val <= current_val <= max_val
            
            # Accuracy percentage per joint angle
            mid_val = (min_val + max_val) / 2.0
            tol = max(15.0, (max_val - min_val) / 2.0)
            diff = abs(current_val - mid_val)

            if diff <= tol:
                joint_score = 100.0
            else:
                joint_score = max(0.0, 100.0 - ((diff - tol) / tol) * 100.0)
                
            joint_scores.append(joint_score)
            
            angle_analysis[angle_name] = {
                "current": round(current_val, 1),
                "target_range": (min_val, max_val),
                "passed": passed,
                "score": round(joint_score, 1)
            }
            
            if not passed:
                all_passed = False
                if current_val < min_val:
                    feedback_messages.append(f"{suggestion} (Increase angle from {round(current_val, 1)}° to {min_val}°)")
                else:
                    feedback_messages.append(f"{suggestion} (Reduce angle from {round(current_val, 1)}° to {max_val}°)")

        overall_accuracy = float(np.mean(joint_scores)) if joint_scores else 100.0

        if all_passed:
            status = "Correct Posture"
            if not feedback_messages:
                feedback_messages.append("Excellent form! Maintain this pose and breathe deeply.")
        else:
            status = "Needs Adjustment"

        return {
            "is_correct": all_passed,
            "accuracy_score": round(overall_accuracy, 1),
            "status": status,
            "feedback": feedback_messages,
            "angle_analysis": angle_analysis
        }
