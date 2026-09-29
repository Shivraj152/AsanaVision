"""
AsanaVision Streamlit Web Application Interface.
Card Mosaic + Asana Library + Earthy & Holistic Yoga Style Interface across 20 Yoga Poses.
"""

import os
import sys
import time
import json
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Include project root in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pose_detection import PoseDetector
from src.feature_extraction import extract_features_from_mediapipe, extract_joint_angles
from src.posture_feedback import PostureEvaluator
from src.predict import PosePredictor
from src.reference_guides import YOGA_POSE_GUIDES

# Page Configuration
st.set_page_config(
    page_title="AsanaVision — Holistic AI Yoga Assistant",
    page_icon="🪷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System: High-Contrast Earthy & Holistic Yoga Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Noto+Sans+Devanagari:wght@600;700&display=swap');
    
    html, body, [class*="css"], div[data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #183B2E !important;
    }
    
    .stApp {
        background-color: #F6F0E4 !important;
        color: #183B2E !important;
    }

    /* Base Typography Defaults */
    .stApp p, .stApp label, .stApp span, .stApp div {
        color: #183B2E;
    }

    /* Top Minimal Earthy Header */
    .header-container {
        background: linear-gradient(135deg, #183B2E 0%, #0F281E 100%) !important;
        padding: 26px 36px !important;
        border-radius: 18px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 12px 30px rgba(24, 59, 46, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        border: 2px solid #245442 !important;
    }
    .header-container h1, 
    .header-container .header-main-title,
    div.header-main-title,
    .header-main-title,
    h1.header-main-title {
        font-family: 'Cinzel', serif !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: 2px !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        text-shadow: 0 3px 10px rgba(0,0,0,0.8) !important;
    }
    .header-container .header-sub-title,
    div.header-sub-title {
        font-size: 1.05rem !important;
        color: #F6F0E4 !important;
        font-weight: 600 !important;
        margin-top: 6px !important;
        opacity: 0.98 !important;
    }
    .header-container .header-status-badge,
    div.header-status-badge {
        background-color: #B8664A !important;
        border: 1.5px solid #E9DDC8 !important;
        color: #FFFFFF !important;
        padding: 8px 18px !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
    }

    /* Hero Banner */
    .hero-banner {
        background-color: #FFFFFF !important;
        border: 1.5px solid #E9DDC8 !important;
        border-radius: 16px !important;
        padding: 24px 28px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 20px rgba(38, 51, 44, 0.05) !important;
    }
    .hero-headline {
        font-family: 'Cinzel', serif !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #183B2E !important;
        margin-bottom: 6px !important;
    }
    .hero-subtext {
        font-size: 1rem !important;
        color: #3A3028 !important;
        font-weight: 600 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FBF9F5 !important;
        border-right: 1.5px solid #E9DDC8 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #183B2E !important;
        font-weight: 700 !important;
    }
    .sidebar-section-header {
        font-family: 'Cinzel', serif !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        color: #183B2E !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-top: 20px !important;
        margin-bottom: 8px !important;
    }

    /* Streamlit Tabs - Ultra High Contrast Fix */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #E9DDC8 !important;
        padding: 8px !important;
        border-radius: 14px !important;
        border: 1.5px solid #DCC9A8 !important;
    }
    .stTabs [data-baseweb="tab"],
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        background-color: #FFFFFF !important;
        border: 1.5px solid #DCC9A8 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        color: #183B2E !important;
        font-weight: 800 !important;
        font-size: 1.02rem !important;
    }
    .stTabs [aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #183B2E !important;
        border: 1.5px solid #183B2E !important;
        box-shadow: 0 4px 12px rgba(24, 59, 46, 0.25) !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Expander Container & Header Overrides */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #E9DDC8 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 15px rgba(24, 59, 46, 0.05) !important;
        overflow: hidden !important;
    }
    div[data-testid="stExpander"] summary {
        background-color: #183B2E !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 14px 20px !important;
    }
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary div,
    div[data-testid="stExpander"] summary label,
    div[data-testid="stExpander"] summary svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        padding: 20px !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] div,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] li {
        color: #183B2E !important;
        font-weight: 600 !important;
    }

    /* Cards */
    .mosaic-card {
        background-color: #FFFFFF !important;
        border: 1.5px solid #E9DDC8 !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 18px rgba(24, 59, 46, 0.06) !important;
        margin-bottom: 20px !important;
    }
    
    /* Accuracy Score Card */
    .accuracy-card-earthy {
        background: linear-gradient(135deg, #F6F0E4 0%, #E9DDC8 100%) !important;
        border: 2px solid #B8664A !important;
        padding: 20px !important;
        border-radius: 16px !important;
        text-align: center !important;
        box-shadow: 0 4px 14px rgba(184, 102, 74, 0.15) !important;
        margin-bottom: 16px !important;
    }
    .accuracy-title-earthy {
        font-family: 'Cinzel', serif !important;
        font-size: 0.95rem !important;
        color: #183B2E !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    .accuracy-score-earthy {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #B8664A !important;
        margin-top: 2px !important;
    }
    
    /* Posture Status Badges */
    .status-badge-sage {
        background-color: #183B2E !important;
        color: #FFFFFF !important;
        border: 1.5px solid #8FAF91 !important;
        padding: 8px 18px !important;
        border-radius: 20px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        display: inline-block !important;
    }
    .status-badge-terracotta {
        background-color: #B8664A !important;
        color: #FFFFFF !important;
        border: 1.5px solid #C7795B !important;
        padding: 8px 18px !important;
        border-radius: 20px !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        display: inline-block !important;
    }

    /* Actionable Coaching Feedback Container */
    .feedback-card-earthy {
        background-color: #FBF9F5 !important;
        border-left: 5px solid #B8664A !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
        margin-top: 14px !important;
        border-top: 1px solid #E9DDC8 !important;
        border-right: 1px solid #E9DDC8 !important;
        border-bottom: 1px solid #E9DDC8 !important;
        color: #183B2E !important;
        font-weight: 700 !important;
    }

    /* Difficulty Badges */
    .difficulty-badge-beginner {
        background-color: #183B2E !important;
        color: #FFFFFF !important;
        padding: 4px 12px !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
    }
    .difficulty-badge-intermediate {
        background-color: #B8664A !important;
        color: #FFFFFF !important;
        padding: 4px 12px !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
    }
    .difficulty-badge-advanced {
        background-color: #991B1B !important;
        color: #FFFFFF !important;
        padding: 4px 12px !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
    }

    /* Sanskrit Title */
    .sanskrit-title {
        font-family: 'Noto Sans Devanagari', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #183B2E !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #183B2E 0%, #0F281E 100%) !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 12px rgba(24, 59, 46, 0.2) !important;
    }
    .stButton > button p,
    .stButton > button span {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #B8664A 0%, #985137 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(184, 102, 74, 0.3) !important;
    }

    /* Form & Input Label Contrast */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span {
        color: #183B2E !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

def open_webcam(device_index: int = 0):
    """
    Robust camera initialization supporting Windows DirectShow (CAP_DSHOW) and multi-index probing.
    """
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            return cap, device_index
        cap.release()

    cap = cv2.VideoCapture(device_index)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            return cap, device_index
        cap.release()

    for idx in range(4):
        if idx == device_index:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                return cap, idx
            cap.release()

        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                return cap, idx
            cap.release()

    return None, device_index

def load_metrics_json():
    metrics_path = "models/model_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGn)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[l[:4] for l in labels],
           yticklabels=[l[:4] for l in labels],
           ylabel='True Pose',
           xlabel='Predicted Pose',
           title='20-Pose Classifier Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    return fig

def process_single_frame(frame, detector, predictor, evaluator, target_pose_override="Any Pose (Auto-detect)"):
    """Processes a single BGR frame, applies pose classification, body visibility gating, and posture evaluation."""
    annotated_frame, results = detector.process_frame(frame)
    annotated_frame = detector.draw_landmarks(annotated_frame, results, draw_bbox=True)

    is_body_vis, vis_msg = detector.check_body_visibility(results)

    if not is_body_vis:
        return annotated_frame, "Body Partial / Out of Frame", 0.0, {
            "is_correct": False,
            "accuracy_score": 0.0,
            "status": "Positioning Required",
            "feedback": [vis_msg, "Please step back 5-6 feet so your full body from shoulders to feet is visible."],
            "angle_analysis": {}
        }, {}

    feature_vector = extract_features_from_mediapipe(results)

    if feature_vector is not None and predictor.is_model_loaded():
        predicted_pose, confidence = predictor.predict(feature_vector, min_confidence=0.50)
        raw_coords = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        angles_dict = extract_joint_angles(np.array(raw_coords))

        eval_pose = target_pose_override if target_pose_override != "Any Pose (Auto-detect)" else predicted_pose

        if "Detecting" in eval_pose or "Uncertain" in eval_pose:
            evaluation = {
                "is_correct": False,
                "accuracy_score": 0.0,
                "status": "Aligning Pose...",
                "feedback": ["Hold your pose steady in front of the camera.", "Ensure good lighting."],
                "angle_analysis": {}
            }
        else:
            evaluation = evaluator.evaluate(eval_pose, angles_dict, confidence)

        annotated_frame = detector.draw_feedback_overlay(
            annotated_frame,
            eval_pose,
            confidence,
            evaluation["is_correct"],
            evaluation["status"],
            evaluation["feedback"]
        )
        return annotated_frame, eval_pose, confidence, evaluation, angles_dict
    else:
        return annotated_frame, "No Person Detected", 0.0, None, {}

def main():
    # Top Minimal Earthy Header Banner
    st.markdown("""
        <div class="header-container">
            <div>
                <div class="header-main-title" style="color: #FFFFFF !important; font-family: 'Cinzel', serif !important; font-size: 2.5rem !important; font-weight: 800 !important; letter-spacing: 2px !important; margin: 0 !important; line-height: 1.2 !important; text-shadow: 0 3px 10px rgba(0,0,0,0.8) !important;">🪷 ASANAVISION</div>
                <div class="header-sub-title" style="color: #F6F0E4 !important; font-size: 1.05rem !important; font-weight: 600 !important; margin-top: 6px !important;">Real-Time Yoga Posture Detection and Feedback Using Machine Learning</div>
            </div>
            <div class="header-status-badge" style="background-color: #B8664A !important; color: #FFFFFF !important; border: 1.5px solid #E9DDC8 !important;">● SYSTEM ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)

    # Hero Introduction Banner
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-headline">Practice with Awareness. Train with AI.</div>
            <div class="hero-subtext">Real-time yoga posture detection and intelligent biomechanical feedback powered by MediaPipe and Scikit-Learn.</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.2rem; font-weight: 800; color: #183B2E; margin-bottom: 12px;">⚙️ CONTROL PANEL</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-section-header">CAMERA HARDWARE</div>', unsafe_allow_html=True)
    camera_device_idx = st.sidebar.number_input("Camera Device Index", min_value=0, max_value=5, value=0)
    
    if st.sidebar.button("🔍 Probe Camera Hardware"):
        with st.sidebar.spinner("Testing camera devices (0-3)..."):
            available_cams = []
            for idx in range(4):
                c, _ = open_webcam(idx)
                if c is not None:
                    available_cams.append(idx)
                    c.release()
            if available_cams:
                st.sidebar.success(f"Active Camera Devices Found: {available_cams}")
            else:
                st.sidebar.error("No available camera found or hardware locked.")

    st.sidebar.markdown('<div class="sidebar-section-header">MODEL & PRACTICE POSE</div>', unsafe_allow_html=True)
    
    available_models = ["Random Forest", "SVM", "KNN", "Logistic Regression"]
    selected_model_name = st.sidebar.selectbox("Select ML Classifier", available_models, index=0)
    
    all_20_poses = sorted(list(YOGA_POSE_GUIDES.keys()))
    selected_pose_guide = st.sidebar.selectbox(
        "Select Practice Yoga Pose (20 Poses)",
        all_20_poses
    )
    
    auto_detect_mode = st.sidebar.checkbox("Auto-detect Any Pose Automatically", value=False)
    target_pose_filter = "Any Pose (Auto-detect)" if auto_detect_mode else selected_pose_guide

    st.sidebar.markdown('<div class="sidebar-section-header">PRACTICE ENVIRONMENT</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem; color: #183B2E; font-weight: 600; line-height: 1.6;">
        <b>Dataset Scope</b>: 20 Yoga Asanas<br>
        <b>Pipeline</b>: OpenCV + MediaPipe + Scikit-Learn<br>
        <b>Architecture</b>: Real-Time Joint Angle Biomechanics
    </div>
    """, unsafe_allow_html=True)

    # Main Navigation Tabs
    tab_live, tab_library, tab_sanskrit, tab_eval, tab_dataset = st.tabs([
        "🌿 Live Practice & Camera",
        "🧘 Asana Library",
        "🕉️ Sanskrit Grid",
        "📊 AI Performance",
        "📚 Academic Info"
    ])

    predictor = PosePredictor(model_path="models/yoga_model.pkl")
    evaluator = PostureEvaluator()

    with tab_live:
        if not predictor.is_model_loaded():
            st.warning("⚠️ Trained model artifact `models/yoga_model.pkl` not found.")
            st.info("Please train the model first by clicking below or running `python scripts/train.py`.")
            
            if st.button("🚀 Train 20-Pose Model Now"):
                with st.spinner("Generating landmarks and training 20-pose model..."):
                    from scripts.generate_synthetic_data import create_dataset
                    from src.train_model import ModelTrainer, SUPPORTED_POSES
                    create_dataset("data/processed/pose_landmarks.csv", samples_per_pose=350, poses=SUPPORTED_POSES)
                    trainer = ModelTrainer("data/processed/pose_landmarks.csv")
                    trainer.train_and_evaluate()
                    trainer.save_model()
                st.success("20-Pose model trained successfully! Please refresh or start camera.")
                st.rerun()

        # Guided Visual Reference Section
        guide_info = YOGA_POSE_GUIDES.get(selected_pose_guide, YOGA_POSE_GUIDES["Tadasana"])
        img_path = guide_info.get("image_file", "")
        sanskrit_txt = guide_info.get("sanskrit_script", selected_pose_guide)
        diff_txt = guide_info.get("difficulty", "Beginner")
        
        diff_badge_class = "difficulty-badge-beginner"
        if diff_txt == "Intermediate":
            diff_badge_class = "difficulty-badge-intermediate"
        elif diff_txt == "Advanced":
            diff_badge_class = "difficulty-badge-advanced"

        with st.expander(f"📖 Visual Practice Guide: {selected_pose_guide} ({guide_info['english_name']}) — {sanskrit_txt}", expanded=True):
            col_guide_img, col_guide_text = st.columns([1, 2])
            
            with col_guide_img:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Target Form: {selected_pose_guide} ({guide_info['english_name']})", use_container_width=True)
                else:
                    st.info("Target pose image loading...")
                
            with col_guide_text:
                st.markdown(f"""
                    <div style='display: flex; align-items: center; gap: 14px; margin-bottom: 8px;'>
                        <span class='sanskrit-title'>{sanskrit_txt}</span>
                        <span style='font-size: 1.3rem; font-weight: 800; color: #183B2E;'>{selected_pose_guide}</span>
                        <span class='{diff_badge_class}'>{diff_txt}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 1rem; color: #183B2E; font-weight: 700; margin-bottom: 8px;'><i>{guide_info['english_name']}</i> ({guide_info['category']})</div>", unsafe_allow_html=True)
                st.markdown("**How to Perform Properly:**")
                for step in guide_info["instructions"]:
                    st.markdown(f"• {step}")
                st.markdown(f"<div style='margin-top: 10px; padding: 12px 16px; background-color: #F6F0E4; border: 1.5px solid #E9DDC8; border-radius: 10px; font-size: 0.95rem; color: #183B2E;'><b>Target Anatomical Angles</b>: <code>{guide_info['target_angles_summary']}</code></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        # Input Mode Selection
        input_mode = st.radio(
            "Camera Input Mode:",
            ["🎥 Live Webcam Stream", "📸 Camera Snapshot (Instant)", "📁 Upload Pose Image"],
            horizontal=True
        )

        col_video, col_info = st.columns([3, 2])

        with col_info:
            st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.25rem; font-weight: 800; color: #183B2E; margin-bottom: 12px;">📊 LIVE POSTURE ANALYTICS</div>', unsafe_allow_html=True)
            
            accuracy_placeholder = st.empty()
            progress_placeholder = st.empty()
            pose_metric_placeholder = st.empty()
            status_metric_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1.5px solid #E9DDC8;'>", unsafe_allow_html=True)
            st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.1rem; font-weight: 800; color: #183B2E; margin-bottom: 10px;">📐 Key Biomechanical Joint Angles</div>', unsafe_allow_html=True)
            angles_placeholder = st.empty()

        with col_video:
            st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.25rem; font-weight: 800; color: #183B2E; margin-bottom: 12px;">📹 LIVE PRACTICE CAMERA FEED</div>', unsafe_allow_html=True)
            frame_placeholder = st.empty()

            if input_mode == "🎥 Live Webcam Stream":
                run_cam = st.toggle("🟢 Activate Live Camera Stream", value=False)
                
                if not run_cam:
                    st.info(f"💡 Switch **'Activate Live Camera Stream'** to ON above to start tracking **{selected_pose_guide}**!")
                    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy_img, f"Target Pose: {selected_pose_guide}", (160, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (143, 175, 145), 2)
                    cv2.putText(dummy_img, "Camera Standby (Toggle ON above)", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (246, 240, 228), 2)
                    frame_placeholder.image(dummy_img, channels="BGR", use_container_width=True)
                    predictor.reset_history()

                else:
                    cap, active_idx = open_webcam(camera_device_idx)
                    
                    if cap is None:
                        st.error(
                            "❌ **Unable to Access Local Server Hardware Webcam**\n\n"
                            "• **Deploying on Streamlit Cloud / Web Server?** Cloud servers do not have physical cameras attached.\n"
                            "  👉 **Please select '📸 Camera Snapshot (Instant)'** above to use your browser or mobile camera directly!\n\n"
                            "• **Running Locally on PC?** Ensure your webcam is connected and not currently in use by Zoom, Teams, or another camera application."
                        )
                        st.info("💡 **Tip:** Switch to **'📸 Camera Snapshot (Instant)'** mode above for live browser camera posture analysis!")
                    else:
                        st.success(f"Connected to Camera Device #{active_idx} | Target Pose: **{selected_pose_guide}**")
                        detector = PoseDetector(static_image_mode=False, min_detection_confidence=0.55)
                        cam_status_placeholder = st.empty()
                        failed_read_count = 0
                        
                        try:
                            while run_cam:
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    failed_read_count += 1
                                    cam_status_placeholder.warning(f"⚠️ Camera read attempt {failed_read_count}/5 failed. Retrying...")
                                    if failed_read_count >= 5:
                                        cam_status_placeholder.error(f"❌ Camera Index {active_idx} is locked by another process.")
                                        break
                                    time.sleep(0.5)
                                    continue

                                failed_read_count = 0
                                cam_status_placeholder.empty()

                                frame = cv2.flip(frame, 1)

                                (
                                    annotated_frame,
                                    predicted_pose,
                                    confidence,
                                    evaluation,
                                    angles_dict
                                ) = process_single_frame(frame, detector, predictor, evaluator, target_pose_filter)

                                acc_score = evaluation.get("accuracy_score", 0.0) if evaluation else 0.0
                                
                                accuracy_placeholder.markdown(
                                    f"""<div class="accuracy-card-earthy">
                                        <div class="accuracy-title-earthy">POSTURE ACCURACY SCORE</div>
                                        <div class="accuracy-score-earthy">{acc_score:.1f}%</div>
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                                progress_placeholder.progress(min(1.0, max(0.0, acc_score / 100.0)))

                                pose_metric_placeholder.metric("Detected / Target Pose", predicted_pose, delta=f"Confidence: {confidence * 100:.1f}%")
                                
                                if evaluation is not None:
                                    status_class = "status-badge-sage" if evaluation.get("is_correct", False) else "status-badge-terracotta"
                                    
                                    status_metric_placeholder.markdown(
                                        f"<b>Posture Alignment Status</b>: <span class='{status_class}'>{evaluation['status']}</span>",
                                        unsafe_allow_html=True
                                    )
                                    fb_text = "<br>".join([f"• {msg}" for msg in evaluation["feedback"]])
                                    feedback_placeholder.markdown(
                                        f"""<div class="feedback-card-earthy">
                                            <b>Actionable Corrective Coaching:</b><br>{fb_text}
                                        </div>""",
                                        unsafe_allow_html=True
                                    )
                                    
                                    if angles_dict:
                                        angle_rows = []
                                        for k, v in angles_dict.items():
                                            match_status = "✅ Perfect" if evaluation.get("angle_analysis", {}).get(k, {}).get("passed", True) else "⚠️ Adjust"
                                            target_r = evaluation.get("angle_analysis", {}).get(k, {}).get("target_range", ("-", "-"))
                                            target_str = f"{target_r[0]}° - {target_r[1]}°" if target_r != ("-", "-") else "-"
                                            
                                            angle_rows.append({
                                                "Joint Angle": k.replace("_", " ").title(),
                                                "Measured": f"{v:.1f}°",
                                                "Target Range": target_str,
                                                "Match": match_status
                                            })
                                        angles_placeholder.table(pd.DataFrame(angle_rows))
                                    else:
                                        angles_placeholder.info("Waiting for full body keypoints...")
                                else:
                                    status_metric_placeholder.markdown("<b>Posture Status</b>: Waiting for body detection...", unsafe_allow_html=True)
                                    feedback_placeholder.info("Step into the camera view so MediaPipe can detect your full body keypoints.")

                                frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                                time.sleep(0.03)

                        finally:
                            cap.release()
                            detector.close()

            elif input_mode == "📸 Camera Snapshot (Instant)":
                img_file_buffer = st.camera_input("Take a photo of your yoga pose")
                if img_file_buffer is not None:
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    detector = PoseDetector(static_image_mode=True, min_detection_confidence=0.55)
                    (
                        annotated_frame,
                        predicted_pose,
                        confidence,
                        evaluation,
                        angles_dict
                    ) = process_single_frame(cv2_img, detector, predictor, evaluator, target_pose_filter)
                    detector.close()

                    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                    acc_score = evaluation.get("accuracy_score", 0.0) if evaluation else 0.0
                    accuracy_placeholder.markdown(
                        f"""<div class="accuracy-card-earthy">
                            <div class="accuracy-title-earthy">POSTURE ACCURACY SCORE</div>
                            <div class="accuracy-score-earthy">{acc_score:.1f}%</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    progress_placeholder.progress(min(1.0, max(0.0, acc_score / 100.0)))
                    pose_metric_placeholder.metric("Detected / Target Pose", predicted_pose, delta=f"Confidence: {confidence * 100:.1f}%")

                    if evaluation is not None:
                        status_class = "status-badge-sage" if evaluation.get("is_correct", False) else "status-badge-terracotta"
                        status_metric_placeholder.markdown(
                            f"<b>Posture Alignment Status</b>: <span class='{status_class}'>{evaluation['status']}</span>",
                            unsafe_allow_html=True
                        )
                        fb_text = "<br>".join([f"• {msg}" for msg in evaluation["feedback"]])
                        feedback_placeholder.markdown(
                            f"""<div class="feedback-card-earthy">
                                <b>Actionable Corrective Coaching:</b><br>{fb_text}
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        if angles_dict:
                            angle_rows = []
                            for k, v in angles_dict.items():
                                match_status = "✅ Perfect" if evaluation.get("angle_analysis", {}).get(k, {}).get("passed", True) else "⚠️ Adjust"
                                target_r = evaluation.get("angle_analysis", {}).get(k, {}).get("target_range", ("-", "-"))
                                target_str = f"{target_r[0]}° - {target_r[1]}°" if target_r != ("-", "-") else "-"
                                
                                angle_rows.append({
                                    "Joint Angle": k.replace("_", " ").title(),
                                    "Measured": f"{v:.1f}°",
                                    "Target Range": target_str,
                                    "Match": match_status
                                })
                            angles_placeholder.table(pd.DataFrame(angle_rows))

            elif input_mode == "📁 Upload Pose Image":
                uploaded_file = st.file_uploader("Choose a pose image (JPG, PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    bytes_data = uploaded_file.read()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                    detector = PoseDetector(static_image_mode=True, min_detection_confidence=0.55)
                    (
                        annotated_frame,
                        predicted_pose,
                        confidence,
                        evaluation,
                        angles_dict
                    ) = process_single_frame(cv2_img, detector, predictor, evaluator, target_pose_filter)
                    detector.close()

                    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                    acc_score = evaluation.get("accuracy_score", 0.0) if evaluation else 0.0
                    accuracy_placeholder.markdown(
                        f"""<div class="accuracy-card-earthy">
                            <div class="accuracy-title-earthy">POSTURE ACCURACY SCORE</div>
                            <div class="accuracy-score-earthy">{acc_score:.1f}%</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    progress_placeholder.progress(min(1.0, max(0.0, acc_score / 100.0)))
                    pose_metric_placeholder.metric("Detected / Target Pose", predicted_pose, delta=f"Confidence: {confidence * 100:.1f}%")

                    if evaluation is not None:
                        status_class = "status-badge-sage" if evaluation.get("is_correct", False) else "status-badge-terracotta"
                        status_metric_placeholder.markdown(
                            f"<b>Posture Alignment Status</b>: <span class='{status_class}'>{evaluation['status']}</span>",
                            unsafe_allow_html=True
                        )
                        fb_text = "<br>".join([f"• {msg}" for msg in evaluation["feedback"]])
                        feedback_placeholder.markdown(
                            f"""<div class="feedback-card-earthy">
                                <b>Actionable Corrective Coaching:</b><br>{fb_text}
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        if angles_dict:
                            angle_rows = []
                            for k, v in angles_dict.items():
                                match_status = "✅ Perfect" if evaluation.get("angle_analysis", {}).get(k, {}).get("passed", True) else "⚠️ Adjust"
                                target_r = evaluation.get("angle_analysis", {}).get(k, {}).get("target_range", ("-", "-"))
                                target_str = f"{target_r[0]}° - {target_r[1]}°" if target_r != ("-", "-") else "-"
                                
                                angle_rows.append({
                                    "Joint Angle": k.replace("_", " ").title(),
                                    "Measured": f"{v:.1f}°",
                                    "Target Range": target_str,
                                    "Match": match_status
                                })
                            angles_placeholder.table(pd.DataFrame(angle_rows))

    with tab_library:
        st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.6rem; font-weight: 800; color: #183B2E; margin-bottom: 8px;">🧘 ASANA LIBRARY MOSAIC</div>', unsafe_allow_html=True)
        st.markdown("Browse yoga poses by difficulty level and explore detailed biomechanical target guidance.")
        
        diff_filter = st.radio("Filter by Difficulty Level:", ["All Poses", "Beginner", "Intermediate", "Advanced"], horizontal=True)
        
        filtered_poses = []
        for name, data in YOGA_POSE_GUIDES.items():
            d_level = data.get("difficulty", "Beginner")
            if diff_filter == "All Poses" or diff_filter == d_level:
                filtered_poses.append((name, data))
                
        # Card Mosaic Layout
        cols = st.columns(3)
        for idx, (pose_name, pose_data) in enumerate(filtered_poses):
            col_target = cols[idx % 3]
            with col_target:
                sansk = pose_data.get("sanskrit_script", pose_name)
                diff = pose_data.get("difficulty", "Beginner")
                d_badge = "difficulty-badge-beginner"
                if diff == "Intermediate":
                    d_badge = "difficulty-badge-intermediate"
                elif diff == "Advanced":
                    d_badge = "difficulty-badge-advanced"
                    
                st.markdown(f"""
                    <div class="mosaic-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div>
                                <div class="sanskrit-title">{sansk}</div>
                                <div style="font-size: 1.2rem; font-weight: 800; color: #183B2E;">{pose_name}</div>
                                <div style="font-size: 0.9rem; color: #183B2E; font-weight: 600;">{pose_data['english_name']}</div>
                            </div>
                            <span class="{d_badge}">{diff}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #183B2E; font-weight: 700; margin-bottom: 10px;">Category: {pose_data['category']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                img_f = pose_data.get("image_file", "")
                if os.path.exists(img_f):
                    st.image(img_f, use_container_width=True)

    with tab_sanskrit:
        st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.6rem; font-weight: 800; color: #183B2E; margin-bottom: 8px;">🕉️ SANSKRIT GRID</div>', unsafe_allow_html=True)
        st.markdown("Traditional Devanagari pose typography grid with English transliteration.")
        
        s_cols = st.columns(4)
        for idx, (pose_name, pose_data) in enumerate(YOGA_POSE_GUIDES.items()):
            col_s = s_cols[idx % 4]
            with col_s:
                st.markdown(f"""
                    <div style="background-color: #FFFFFF; border: 1.5px solid #E9DDC8; border-radius: 14px; padding: 18px; text-align: center; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(24, 59, 46, 0.05);">
                        <div style="font-family: 'Noto Sans Devanagari', sans-serif; font-size: 1.9rem; font-weight: 800; color: #B8664A; margin-bottom: 4px;">{pose_data.get('sanskrit_script', pose_name)}</div>
                        <div style="font-family: 'Cinzel', serif; font-size: 1.1rem; font-weight: 800; color: #183B2E;">{pose_name}</div>
                        <div style="font-size: 0.88rem; color: #183B2E; font-weight: 600;">{pose_data['english_name']}</div>
                    </div>
                """, unsafe_allow_html=True)

    with tab_eval:
        st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.5rem; font-weight: 800; color: #183B2E; margin-bottom: 8px;">📊 Model Evaluation & Classifier Comparison</div>', unsafe_allow_html=True)
        st.markdown("Measured performance metrics across scikit-learn Machine Learning models trained on normalized MediaPipe landmark features.")

        metrics_data = load_metrics_json()

        if metrics_data:
            summary_rows = []
            for name, m in metrics_data.items():
                summary_rows.append({
                    "Model Algorithm": name,
                    "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                    "Precision": f"{m['precision'] * 100:.2f}%",
                    "Recall": f"{m['recall'] * 100:.2f}%",
                    "F1-Score": f"{m['f1_score'] * 100:.2f}%"
                })

            st.markdown('<div style="font-size: 1.1rem; font-weight: 800; color: #183B2E; margin-top: 15px; margin-bottom: 10px;">Model Performance Comparison (20 Yoga Poses)</div>', unsafe_allow_html=True)
            st.table(pd.DataFrame(summary_rows))

            st.markdown("<hr style='margin: 20px 0; border: none; border-top: 1.5px solid #E9DDC8;'>", unsafe_allow_html=True)
            col_cm1, col_cm2 = st.columns(2)

            with col_cm1:
                st.markdown('<div style="font-size: 1.05rem; font-weight: 800; color: #183B2E; margin-bottom: 10px;">Random Forest Confusion Matrix</div>', unsafe_allow_html=True)
                rf_m = metrics_data.get("Random Forest")
                if rf_m:
                    fig_rf = plot_confusion_matrix(np.array(rf_m["confusion_matrix"]), rf_m["labels"])
                    st.pyplot(fig_rf)

            with col_cm2:
                st.markdown('<div style="font-size: 1.05rem; font-weight: 800; color: #183B2E; margin-bottom: 10px;">Support Vector Machine (SVM) Confusion Matrix</div>', unsafe_allow_html=True)
                svm_m = metrics_data.get("SVM")
                if svm_m:
                    fig_svm = plot_confusion_matrix(np.array(svm_m["confusion_matrix"]), svm_m["labels"])
                    st.pyplot(fig_svm)
        else:
            st.info("Model evaluation metrics JSON not found. Train the model via `python scripts/train.py` to populate performance data.")

    with tab_dataset:
        st.markdown('<div style="font-family: \'Cinzel\', serif; font-size: 1.5rem; font-weight: 800; color: #183B2E; margin-bottom: 8px;">📚 Dataset & Academic References</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Yoga Asana Dataset Specification (20 Distinct Poses)
        This project expands the reference architecture defined in:
        
        > **Paper**: *Yoga Asana: A Dataset for Yoga Pose Recognition and Correctness Assessment*  
        > **Scope**: 20 Yoga Asanas | Complete Visual Reference Guides | Real-Time Feedback & Camera Optimization  
        
        #### Supported Asana Classes (20 Poses):
        1. **Anantasana** (Side-Reclining Leg Lift)
        2. **Ardhakati Chakrasana** (Standing Side Bend)
        3. **Balasana** (Child's Pose)
        4. **Bhujangasana** (Cobra Pose)
        5. **Halasana** (Plow Pose)
        6. **Kati Chakrasana** (Standing Spinal Twist)
        7. **Marjariasana** (Cat-Cow Pose)
        8. **Parvatasana** (Downward Mountain Pose)
        9. **Paschimottanasana** (Seated Forward Bend)
        10. **Sarvangasana** (Shoulderstand)
        11. **Setu Bandhasana** (Bridge Pose)
        12. **Tadasana** (Mountain Pose)
        13. **Trikonasana** (Triangle Pose)
        14. **Ustrasana** (Camel Pose)
        15. **Utkatasana** (Chair Pose)
        16. **Vajrasana** (Thunderbolt Pose)
        17. **Viparita Karani** (Legs-up-the-wall Pose)
        18. **Virabhadrasana I** (Warrior I Pose)
        19. **Virabhadrasana II** (Warrior II Pose)
        20. **Vrikshasana** (Tree Pose)
        
        #### Repository Download Sources:
        - **Zenodo Repository**: [https://doi.org/10.5281/zenodo.7818789](https://doi.org/10.5281/zenodo.7818789)
        - **Mendeley Data**: [https://data.mendeley.com/datasets/jc4mmnvcdk1](https://data.mendeley.com/datasets/jc4mmnvcdk1)
        """)

if __name__ == "__main__":
    main()
