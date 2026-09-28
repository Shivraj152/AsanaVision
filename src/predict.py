"""
Real-time pose prediction and model inference wrapper for AsanaVision.
Loads serialized scikit-learn model, applies weighted temporal prediction smoothing, and handles low confidence gating across 20 yoga poses.
"""

import os
import joblib
import numpy as np
from collections import deque, Counter
from typing import Tuple, Dict, Any, Optional

class PosePredictor:
    """Wrapper class for model loading, real-time inference, and temporal smoothing."""

    def __init__(self, model_path: str = "models/yoga_model.pkl", buffer_size: int = 7):
        self.model_path = model_path
        self.model = None
        self.classes = []
        self.history = deque(maxlen=buffer_size)
        self._load_model()

    def _load_model(self):
        """Loads model file from disk with robust error checking."""
        if not os.path.exists(self.model_path):
            self.model = None
            return

        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "classes_"):
                self.classes = list(self.model.classes_)
            elif hasattr(self.model, "named_steps"):
                last_step = list(self.model.named_steps.values())[-1]
                if hasattr(last_step, "classes_"):
                    self.classes = list(last_step.classes_)
        except Exception:
            self.model = None

    def is_model_loaded(self) -> bool:
        """Returns True if trained model is loaded and ready for prediction."""
        return self.model is not None

    def reset_history(self):
        """Clears prediction temporal buffer."""
        self.history.clear()

    def predict(self, feature_vector: np.ndarray, min_confidence: float = 0.50) -> Tuple[str, float]:
        """
        Predicts yoga pose class with temporal smoothing and confidence gating.

        :param feature_vector: 1D array of 109 floats (99 normalized coordinates + 10 angles).
        :param min_confidence: Minimum probability threshold for a valid prediction.
        :return: Tuple of (predicted pose label, confidence score between 0.0 and 1.0).
        """
        if not self.is_model_loaded():
            return "Unknown (Model Not Trained)", 0.0

        if feature_vector is None or len(feature_vector) == 0:
            self.history.clear()
            return "No Person Detected", 0.0

        # Reshape for single sample prediction
        sample = feature_vector.reshape(1, -1)

        try:
            prediction = str(self.model.predict(sample)[0])

            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(sample)[0]
                confidence = float(np.max(probas))
            else:
                confidence = 1.0

            if confidence < min_confidence:
                return "Detecting / Aligning Pose", confidence

            self.history.append((prediction, confidence))

            # Majority voting across temporal buffer
            labels = [p[0] for p in self.history]
            most_common_label = Counter(labels).most_common(1)[0][0]

            matching_confidences = [c for l, c in self.history if l == most_common_label]
            avg_confidence = float(np.mean(matching_confidences))

            return most_common_label, avg_confidence

        except Exception as e:
            return f"Inference Error: {str(e)}", 0.0
