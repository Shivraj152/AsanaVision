"""
MediaPipe Pose Detection wrapper for AsanaVision.
Provides robust real-time human pose detection, landmark extraction, performance optimization, and skeleton rendering.
"""

import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

class PoseDetector:
    """Wrapper class around MediaPipe Pose estimation engine optimized for high FPS and camera accuracy."""
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55
    ):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame: np.ndarray, target_max_dim: int = 720) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Processes an OpenCV BGR frame and returns annotated frame and MediaPipe pose landmarks.
        Includes fast frame scaling optimization for enhanced camera FPS and detection speed.
        
        :param frame: Input image in OpenCV BGR format.
        :param target_max_dim: Maximum dimension for real-time camera processing efficiency.
        :return: Tuple of (processed/annotated frame, pose_landmarks result object)
        """
        if frame is None or frame.size == 0:
            return frame, None

        # Efficiency Optimization: Optional fast resize if camera resolution is excessively large
        h, w = frame.shape[:2]
        if max(h, w) > target_max_dim and not self.pose._static_image_mode:
            scale = target_max_dim / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        try:
            results = self.pose.process(image_rgb)
        except Exception:
            return frame, None
            
        image_rgb.flags.writeable = True

        return frame, results

    def draw_landmarks(
        self,
        frame: np.ndarray,
        results: Any,
        draw_bbox: bool = False
    ) -> np.ndarray:
        """
        Draws 33 pose landmarks and skeleton connections on the frame.
        
        :param frame: Image frame (BGR).
        :param results: MediaPipe pose process output.
        :param draw_bbox: If True, draws a bounding box around detected person.
        :return: Frame with drawn landmarks.
        """
        annotated_frame = frame.copy()
        if results is None or not results.pose_landmarks:
            return annotated_frame

        # Draw pose connections and landmarks with custom style
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        if draw_bbox and results.pose_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in results.pose_landmarks.landmark]
            y_coords = [lm.y * h for lm in results.pose_landmarks.landmark]
            
            x_min, x_max = int(max(0, min(x_coords))), int(min(w, max(x_coords)))
            y_min, y_max = int(max(0, min(y_coords))), int(min(h, max(y_coords)))
            
            # Padding around bounding box
            pad_x = int((x_max - x_min) * 0.05)
            pad_y = int((y_max - y_min) * 0.05)
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)

            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 220, 100), 2)
            cv2.putText(
                annotated_frame,
                "Person Detected",
                (x_min, max(15, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 100),
                2
            )

        return annotated_frame

    def extract_raw_landmarks(self, results: Any) -> Optional[List[float]]:
        """
        Extracts raw 33 landmark points (x, y, z, visibility) as a flat 132-element list.
        """
        if results is None or not results.pose_landmarks:
            return None

        raw_landmarks = []
        for lm in results.pose_landmarks.landmark:
            raw_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            
        return raw_landmarks

    def draw_feedback_overlay(
        self,
        frame: np.ndarray,
        pose_name: str,
        confidence: float,
        is_correct: bool,
        status_text: str,
        feedback_msgs: List[str]
    ) -> np.ndarray:
        """
        Renders a sleek real-time HUD banner directly on the video frame.
        """
        annotated = frame.copy()
        h, w, _ = frame.shape
        
        # Header banner rectangle background
        banner_h = 75
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (20, 20, 20), -1)
        
        # Status color in BGR (Green for correct, Red for adjustment)
        status_color = (60, 210, 80) if is_correct else (60, 60, 230)
        
        # Pose Name & Confidence
        cv2.putText(annotated, f"POSE: {pose_name.upper()} ({confidence*100:.1f}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(annotated, f"STATUS: {status_text.upper()}", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
        # Bottom feedback banner if feedback exists
        if feedback_msgs:
            visible_msgs = feedback_msgs[:3]
            bot_h = 30 + len(visible_msgs) * 25
            cv2.rectangle(annotated, (0, h - bot_h), (w, h), (20, 20, 20), -1)
            for i, msg in enumerate(visible_msgs):
                cv2.putText(annotated, f"> {msg}", (15, h - bot_h + 22 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                            
        return annotated

    def check_body_visibility(self, results: Any, min_vis: float = 0.40) -> Tuple[bool, str]:
        """
        Verifies if core keypoints (shoulders, hips, knees, ankles) are visible in the frame.
        """
        if results is None or not results.pose_landmarks:
            return False, "No person detected in camera view."
            
        landmarks = results.pose_landmarks.landmark
        key_indices = [11, 12, 23, 24, 25, 26, 27, 28] # Shoulders, hips, knees, ankles
        
        low_vis_count = 0
        for idx in key_indices:
            if landmarks[idx].visibility < min_vis:
                low_vis_count += 1
                
        if low_vis_count > 3:
            return False, "Step back so your full body (shoulders to feet) is visible in frame."
            
        return True, "Full Body Visible"

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
