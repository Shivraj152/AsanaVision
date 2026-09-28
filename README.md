# AsanaVision: Real-Time Yoga Posture Detection and Feedback Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green.svg)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

**AsanaVision** is an academic real-time computer vision and machine learning application that detects yoga body poses via a local webcam, computes scale-invariant 3D keypoint features, classifies 10 distinct yoga asanas using scikit-learn ML models, evaluates biomechanical posture correctness, and provides real-time corrective feedback through a Streamlit dashboard.

---

## Technical Features & Computer Vision Pipeline

```text
Webcam Stream
   │
   ▼
OpenCV Frame Capture (BGR -> RGB)
   │
   ▼
MediaPipe Pose Estimation (33 3D Body Landmarks)
   │
   ▼
Feature Extraction & Normalization
 (Pelvis Centering + Torso Scale Normalization + 10 Biomechanical Joint Angles)
   │
   ▼
Machine Learning Classifier (Random Forest / SVM / KNN / Logistic Regression)
   │
   ▼
Yoga Pose Prediction & Confidence Assessment
   │
   ▼
Biomechanical Rule Evaluation Engine (Joint angle deviation analysis)
   │
   ▼
Streamlit Interactive Dashboard (Live Skeleton Feed + Metrics + Real-Time Feedback)
```

---

## Technology Stack

- **Core Language**: Python (3.9 - 3.11)
- **Computer Vision**: OpenCV (`opencv-python`), MediaPipe Pose (`mediapipe`)
- **Data Engineering & Math**: NumPy, Pandas
- **Machine Learning**: scikit-learn, Joblib
- **Web Dashboard**: Streamlit
- **Visualization**: Matplotlib

---

## Supported Yoga Asanas (10 Classes)

1. **Anantasana** (*Side-Reclining Leg Lift*)
2. **Ardhakati Chakrasana** (*Standing Side Bend*)
3. **Bhujangasana** (*Cobra Pose*)
4. **Kati Chakrasana** (*Standing Spinal Twist*)
5. **Marjariasana** (*Cat-Cow Pose*)
6. **Parvatasana** (*Downward Mountain Pose*)
7. **Sarvangasana** (*Shoulderstand*)
8. **Tadasana** (*Mountain Pose*)
9. **Vajrasana** (*Thunderbolt Pose*)
10. **Viparita Karani** (*Legs-Up-The-Wall Pose*)

*Initial Prototype Mode supports the core 3 poses:* **Tadasana**, **Bhujangasana**, **Vajrasana**.

---

## Dataset References & Specification

This implementation adheres to the dataset specification described in:

- **Paper Title**: *Yoga Asana: A Dataset for Yoga Pose Recognition and Correctness Assessment*
- **Scope**: 10 Yoga Asanas | ~11,344 Images | 80 Videos | Right/Correct and Wrong/Incorrect posture examples

### Download Sources
- **Zenodo Repository**: [https://doi.org/10.5281/zenodo.7818789](https://doi.org/10.5281/zenodo.7818789)
- **Mendeley Data**: [https://data.mendeley.com/datasets/jc4mmnvcdk1](https://data.mendeley.com/datasets/jc4mmnvcdk1)

---

## HOW TO RUN LOCALLY (Windows CMD Instructions)

### Prerequisites & Environment Setup

Open Windows Command Prompt (`cmd.exe`) or PowerShell and navigate to the project directory:

```cmd
cd d:\AsanaVision
```

1. **Create Virtual Environment**:
   ```cmd
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   ```cmd
   venv\Scripts\activate
   ```

3. **Upgrade Pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

4. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

---

### Step-by-Step Workflow Commands

#### 1. Preparing the Dataset
- **Option A (Instant Local Bootstrapping - Recommended)**:
  Generate realistic landmark samples for all 10 poses:
  ```cmd
  python scripts/generate_synthetic_data.py --samples 300
  ```
  *(To test the initial 3-pose prototype only, pass `--initial_only`)*.

- **Option B (Using Downloaded Raw Images)**:
  Extract images from Zenodo/Mendeley into `data/raw/<Pose_Name>/*.jpg` folders, then run:
  ```cmd
  python scripts/extract_landmarks.py --raw_dir data/raw --output data/processed/pose_landmarks.csv
  ```

#### 2. Training and Comparing Models
Execute the training pipeline to train Random Forest, SVM, KNN, and Logistic Regression, display evaluation metrics, print the confusion matrix, and save `models/yoga_model.pkl`:

```cmd
python scripts/train.py
```

*(For initial 3-pose prototype training only, run `python scripts/train.py --initial_only`)*.

#### 3. Running Unit Tests
Verify pose normalization, angle math, and feedback engine logic:

```cmd
python -m unittest discover tests
```

#### 4. Starting the Streamlit Application
Launch the web dashboard:

```cmd
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## FIRST TEST

To perform your first end-to-end verification, run the following command in your terminal:

```cmd
python -m unittest discover tests
```

**Expected Successful Output:**
```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.005s

OK
```

Followed by running model evaluation:

```cmd
python scripts/train.py
```

**Expected Output:**
```text
========================================================
       ASANAVISION MODEL TRAINING & EVALUATION          
========================================================

Classifier             | Accuracy   | Precision  | Recall     | F1-Score  
---------------------------------------------------------------------------
Random Forest          | 100.00%    | 100.00%    | 100.00%    | 100.00%   
SVM                    | 100.00%    | 100.00%    | 100.00%    | 100.00%   
KNN                    | 90.00%     | 81.84%     | 90.00%     | 85.51%    
Logistic Regression    | 100.00%    | 100.00%    | 100.00%    | 100.00%   
---------------------------------------------------------------------------

Saved primary classifier model to: models\yoga_model.pkl
Saved evaluation metrics JSON to: models\model_metrics.json
```

---

## TROUBLESHOOTING

| `Error: Found app.py but it does not export a top-level "app" (Vercel)` | Vercel expects Flask/WSGI apps, not Streamlit apps | Deploy on **Streamlit Community Cloud** (share.streamlit.io) or **Hugging Face Spaces** which natively host Streamlit long-running web servers. |
| `FileNotFoundError: models/yoga_model.pkl` | Model file has not been trained yet | Run `python scripts/train.py` or click the "Train Model Now" button inside the Streamlit dashboard tab. |
| `ModuleNotFoundError: No module named 'src'` | Python execution path issue | Always run scripts from the project root or run `pip install -e .` |
| `Port 8501 is already in use` | Existing Streamlit process active | Run `streamlit run app.py --server.port 8502` to use a different port. |
| `MediaPipe processing error / DLL load failed` | Outdated MSVC redistributable or missing OpenCV dependencies | Install Visual C++ Redistributable 2015-2022 and update MediaPipe: `pip install --upgrade mediapipe opencv-python`. |

---

## Academic References & Citations

1. **Project Title Reference**:
   *“AsanaVision: Real-Time Yoga Posture Detection and Feedback Using Machine Learning”*
2. **Dataset Paper Reference**:
   *Yoga Asana: A Dataset for Yoga Pose Recognition and Correctness Assessment*
3. **Zenodo Dataset**:
   [https://doi.org/10.5281/zenodo.7818789](https://doi.org/10.5281/zenodo.7818789)
4. **Mendeley Data**:
   [https://data.mendeley.com/datasets/jc4mmnvcdk1](https://data.mendeley.com/datasets/jc4mmnvcdk1)
5. **MediaPipe Documentation**:
   [https://developers.google.com/mediapipe/solutions/vision/pose_landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
6. **scikit-learn Documentation**:
   [https://scikit-learn.org](https://scikit-learn.org)
