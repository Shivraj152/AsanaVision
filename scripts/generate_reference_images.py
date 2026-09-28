"""
Generates high quality 1376x768 JPG reference guide illustrations for the 10 new yoga poses in data/reference_images/.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REFERENCE_DIR = "data/reference_images"
os.makedirs(REFERENCE_DIR, exist_ok=True)

NEW_POSES = {
    "Vrikshasana.jpg": {
        "title": "Vrikshasana (Tree Pose)",
        "category": "Standing Balance",
        "angles": "Standing Leg: 180° | Bent Knee: ~45° | Torso: Vertical",
        "draw_func": lambda draw, w, h: draw_tree_pose(draw, w, h)
    },
    "Trikonasana.jpg": {
        "title": "Trikonasana (Triangle Pose)",
        "category": "Standing Lateral Stretch",
        "angles": "Both Knees: 180° | Arms: Extended T | Torso Tilt: ~50°",
        "draw_func": lambda draw, w, h: draw_triangle_pose(draw, w, h)
    },
    "Virabhadrasana_I.jpg": {
        "title": "Virabhadrasana I (Warrior I)",
        "category": "Standing Lunge",
        "angles": "Front Knee: 90° | Back Leg: 180° | Arms: Overhead",
        "draw_func": lambda draw, w, h: draw_warrior_1(draw, w, h)
    },
    "Virabhadrasana_II.jpg": {
        "title": "Virabhadrasana II (Warrior II)",
        "category": "Standing Wide Lunge",
        "angles": "Front Knee: 90° | Back Leg: 180° | Arms: T-Shape 180°",
        "draw_func": lambda draw, w, h: draw_warrior_2(draw, w, h)
    },
    "Utkatasana.jpg": {
        "title": "Utkatasana (Chair Pose)",
        "category": "Standing Power Squat",
        "angles": "Knees: Bent ~100° | Torso: Slanted ~35° | Arms: High Overhead",
        "draw_func": lambda draw, w, h: draw_chair_pose(draw, w, h)
    },
    "Setu_Bandhasana.jpg": {
        "title": "Setu Bandhasana (Bridge Pose)",
        "category": "Supine Backbend",
        "angles": "Hips: Lifted Arch ~150° | Knees: Bent ~90°",
        "draw_func": lambda draw, w, h: draw_bridge_pose(draw, w, h)
    },
    "Balasana.jpg": {
        "title": "Balasana (Child's Pose)",
        "category": "Restorative Kneeling Fold",
        "angles": "Knees: Fully Folded <40° | Hips: Folded | Arms: Forward",
        "draw_func": lambda draw, w, h: draw_child_pose(draw, w, h)
    },
    "Paschimottanasana.jpg": {
        "title": "Paschimottanasana (Seated Fold)",
        "category": "Seated Forward Bend",
        "angles": "Knees: Extended 180° | Hip Fold: Deep ~40°",
        "draw_func": lambda draw, w, h: draw_forward_fold(draw, w, h)
    },
    "Ustrasana.jpg": {
        "title": "Ustrasana (Camel Pose)",
        "category": "Kneeling Backbend",
        "angles": "Torso Backbend: Arched ~50° | Knees: Kneeling 90°",
        "draw_func": lambda draw, w, h: draw_camel_pose(draw, w, h)
    },
    "Halasana.jpg": {
        "title": "Halasana (Plow Pose)",
        "category": "Inverted Forward Fold",
        "angles": "Legs: Overhead 180° | Hip Inversion Fold: ~60°",
        "draw_func": lambda draw, w, h: draw_plow_pose(draw, w, h)
    }
}

def create_studio_backdrop(width=1376, height=768):
    # Radial dark studio background
    img = Image.new("RGB", (width, height), "#0F172A")
    draw = ImageDraw.Draw(img)
    
    # Subtle gradient studio floor glow
    for r in range(height, 0, -5):
        alpha = int(255 * (r / height))
        color = (15 + int(15 * (1 - r/height)), 23 + int(25 * (1 - r/height)), 42 + int(40 * (1 - r/height)))
        draw.rectangle([0, height - r, width, height], fill=color)
        
    # Grid lines floor effect
    for y in range(500, height, 40):
        draw.line([(0, y), (width, y)], fill="#1E293B", width=1)
    for x in range(0, width, 80):
        draw.line([(x, 500), (width//2 + (x - width//2)*2, height)], fill="#1E293B", width=1)
        
    return img

def draw_header_card(draw, title, category, angles, width=1376):
    # Header card background panel
    draw.rounded_rectangle([40, 40, width - 40, 130], radius=15, fill="#1E293B", outline="#38BDF8", width=2)
    draw.text((60, 52), f"ASANAVISION VISUAL REFERENCE GUIDE", fill="#38BDF8")
    draw.text((60, 78), title, fill="#FFFFFF")
    draw.text((width - 450, 60), f"Category: {category}", fill="#4ADE80")
    draw.text((width - 450, 90), f"Target Angles: {angles}", fill="#94A3B8")

def draw_joint(draw, pt, radius=10, fill="#38BDF8", outline="#FFFFFF"):
    draw.ellipse([pt[0]-radius, pt[1]-radius, pt[0]+radius, pt[1]+radius], fill=fill, outline=outline, width=2)

def draw_bone(draw, p1, p2, width=8, color="#38BDF8"):
    draw.line([p1, p2], fill=color, width=width)

# Drawing skeletal figures per pose
def draw_tree_pose(draw, w, h):
    cx, cy = w // 2, 430
    head = (cx, cy - 220)
    neck = (cx, cy - 180)
    l_sh = (cx - 40, cy - 170)
    r_sh = (cx + 40, cy - 170)
    elbow_l = (cx - 70, cy - 250)
    elbow_r = (cx + 70, cy - 250)
    hands = (cx, cy - 290)
    hip_l = (cx - 25, cy - 40)
    hip_r = (cx + 25, cy - 40)
    knee_l = (cx - 25, cy + 80)
    ankle_l = (cx - 25, cy + 200)
    knee_r = (cx + 100, cy + 30)
    foot_r = (cx + 25, cy + 70)
    
    # Bones
    draw_bone(draw, hands, elbow_l, 6, "#4ADE80")
    draw_bone(draw, hands, elbow_r, 6, "#4ADE80")
    draw_bone(draw, elbow_l, l_sh, 6, "#4ADE80")
    draw_bone(draw, elbow_r, r_sh, 6, "#4ADE80")
    draw_bone(draw, l_sh, r_sh, 8, "#38BDF8")
    draw_bone(draw, (cx, cy-170), (cx, cy-40), 10, "#38BDF8")
    draw_bone(draw, hip_l, hip_r, 8, "#38BDF8")
    draw_bone(draw, hip_l, knee_l, 9, "#38BDF8")
    draw_bone(draw, knee_l, ankle_l, 9, "#38BDF8")
    draw_bone(draw, hip_r, knee_r, 9, "#4ADE80")
    draw_bone(draw, knee_r, foot_r, 9, "#4ADE80")
    
    # Head & Joints
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [l_sh, r_sh, hip_l, hip_r, knee_l, ankle_l, knee_r]:
        draw_joint(draw, pt, 8)

def draw_triangle_pose(draw, w, h):
    cx, cy = w // 2, 400
    head = (cx + 120, cy - 60)
    l_sh = (cx + 30, cy - 40)
    r_sh = (cx + 90, cy - 10)
    arm_up = (cx + 170, cy - 160)
    arm_down = (cx - 30, cy + 110)
    hip_l = (cx - 60, cy + 10)
    hip_r = (cx, cy + 30)
    knee_l = (cx - 160, cy + 130)
    ankle_l = (cx - 240, cy + 220)
    knee_r = (cx + 60, cy + 130)
    ankle_r = (cx + 120, cy + 220)
    
    draw_bone(draw, arm_up, r_sh, 8, "#4ADE80")
    draw_bone(draw, r_sh, l_sh, 8, "#4ADE80")
    draw_bone(draw, l_sh, arm_down, 8, "#4ADE80")
    draw_bone(draw, l_sh, hip_l, 10, "#38BDF8")
    draw_bone(draw, r_sh, hip_r, 10, "#38BDF8")
    draw_bone(draw, hip_l, knee_l, 9, "#38BDF8")
    draw_bone(draw, knee_l, ankle_l, 9, "#38BDF8")
    draw_bone(draw, hip_r, knee_r, 9, "#38BDF8")
    draw_bone(draw, knee_r, ankle_r, 9, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [l_sh, r_sh, hip_l, hip_r, knee_l, ankle_l, knee_r, ankle_r, arm_up, arm_down]:
        draw_joint(draw, pt, 8)

def draw_warrior_1(draw, w, h):
    cx, cy = w // 2, 420
    head = (cx + 40, cy - 230)
    l_sh = (cx + 15, cy - 170)
    r_sh = (cx + 65, cy - 170)
    hands = (cx + 40, cy - 300)
    hip_l = (cx - 40, cy - 40)
    hip_r = (cx + 40, cy - 40)
    knee_front = (cx + 160, cy + 30)
    ankle_front = (cx + 160, cy + 190)
    knee_back = (cx - 130, cy + 70)
    ankle_back = (cx - 210, cy + 190)
    
    draw_bone(draw, hands, (cx+40, cy-170), 8, "#4ADE80")
    draw_bone(draw, l_sh, r_sh, 8, "#38BDF8")
    draw_bone(draw, (cx+40, cy-170), (cx, cy-40), 10, "#38BDF8")
    draw_bone(draw, hip_r, knee_front, 9, "#4ADE80")
    draw_bone(draw, knee_front, ankle_front, 9, "#4ADE80")
    draw_bone(draw, hip_l, knee_back, 9, "#38BDF8")
    draw_bone(draw, knee_back, ankle_back, 9, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [l_sh, r_sh, hip_l, hip_r, knee_front, ankle_front, knee_back, ankle_back, hands]:
        draw_joint(draw, pt, 8)

def draw_warrior_2(draw, w, h):
    cx, cy = w // 2, 430
    head = (cx, cy - 200)
    l_sh = (cx - 40, cy - 150)
    r_sh = (cx + 40, cy - 150)
    arm_front = (cx + 220, cy - 150)
    arm_back = (cx - 220, cy - 150)
    hip_l = (cx - 30, cy - 30)
    hip_r = (cx + 30, cy - 30)
    knee_front = (cx + 150, cy + 30)
    ankle_front = (cx + 150, cy + 180)
    knee_back = (cx - 140, cy + 70)
    ankle_back = (cx - 220, cy + 180)
    
    draw_bone(draw, arm_back, arm_front, 8, "#4ADE80")
    draw_bone(draw, (cx, cy-150), (cx, cy-30), 10, "#38BDF8")
    draw_bone(draw, hip_r, knee_front, 9, "#4ADE80")
    draw_bone(draw, knee_front, ankle_front, 9, "#4ADE80")
    draw_bone(draw, hip_l, knee_back, 9, "#38BDF8")
    draw_bone(draw, knee_back, ankle_back, 9, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [l_sh, r_sh, hip_l, hip_r, knee_front, ankle_front, knee_back, ankle_back, arm_front, arm_back]:
        draw_joint(draw, pt, 8)

def draw_chair_pose(draw, w, h):
    cx, cy = w // 2 - 50, 420
    head = (cx - 30, cy - 220)
    l_sh = (cx - 50, cy - 170)
    hands = (cx - 110, cy - 290)
    hip = (cx + 50, cy - 30)
    knee = (cx - 60, cy + 70)
    ankle = (cx - 30, cy + 200)
    
    draw_bone(draw, hands, l_sh, 8, "#4ADE80")
    draw_bone(draw, l_sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 10, "#38BDF8")
    draw_bone(draw, knee, ankle, 10, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [l_sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def draw_bridge_pose(draw, w, h):
    cx, cy = w // 2, 450
    head = (cx - 260, cy + 60)
    sh = (cx - 210, cy + 40)
    hip = (cx - 20, cy - 80)
    knee = (cx + 150, cy + 10)
    ankle = (cx + 160, cy + 140)
    hands = (cx - 50, cy + 60)
    
    draw_bone(draw, sh, hands, 8, "#4ADE80")
    draw_bone(draw, sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 10, "#38BDF8")
    draw_bone(draw, knee, ankle, 10, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def draw_child_pose(draw, w, h):
    cx, cy = w // 2, 480
    head = (cx + 160, cy + 20)
    sh = (cx + 100, cy - 20)
    hip = (cx - 100, cy - 40)
    knee = (cx - 120, cy + 60)
    ankle = (cx - 180, cy + 60)
    hands = (cx + 250, cy + 30)
    
    draw_bone(draw, sh, hands, 8, "#4ADE80")
    draw_bone(draw, sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 9, "#38BDF8")
    draw_bone(draw, knee, ankle, 9, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def draw_forward_fold(draw, w, h):
    cx, cy = w // 2, 460
    head = (cx + 120, cy - 40)
    sh = (cx + 60, cy - 50)
    hip = (cx - 120, cy - 30)
    knee = (cx, cy - 20)
    ankle = (cx + 180, cy - 10)
    hands = (cx + 170, cy - 30)
    
    draw_bone(draw, sh, hands, 8, "#4ADE80")
    draw_bone(draw, sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 10, "#38BDF8")
    draw_bone(draw, knee, ankle, 10, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def draw_camel_pose(draw, w, h):
    cx, cy = w // 2, 430
    head = (cx - 90, cy - 180)
    sh = (cx - 50, cy - 130)
    hip = (cx + 30, cy - 40)
    knee = (cx + 30, cy + 120)
    ankle = (cx - 80, cy + 120)
    hands = (cx - 80, cy + 100)
    
    draw_bone(draw, sh, hands, 8, "#4ADE80")
    draw_bone(draw, sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 10, "#38BDF8")
    draw_bone(draw, knee, ankle, 10, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def draw_plow_pose(draw, w, h):
    cx, cy = w // 2, 440
    head = (cx + 150, cy + 40)
    sh = (cx + 100, cy + 30)
    hip = (cx + 20, cy - 90)
    knee = (cx - 110, cy - 30)
    ankle = (cx - 200, cy + 30)
    hands = (cx + 20, cy + 40)
    
    draw_bone(draw, sh, hands, 8, "#4ADE80")
    draw_bone(draw, sh, hip, 10, "#38BDF8")
    draw_bone(draw, hip, knee, 10, "#38BDF8")
    draw_bone(draw, knee, ankle, 10, "#38BDF8")
    draw_joint(draw, head, 22, "#38BDF8")
    for pt in [sh, hip, knee, ankle, hands]:
        draw_joint(draw, pt, 8)

def generate_all():
    width, height = 1376, 768
    for filename, info in NEW_POSES.items():
        img = create_studio_backdrop(width, height)
        draw = ImageDraw.Draw(img)
        
        # Draw header card
        draw_header_card(draw, info["title"], info["category"], info["angles"], width)
        
        # Draw pose artwork
        info["draw_func"](draw, width, height)
        
        # Save output
        out_path = os.path.join(REFERENCE_DIR, filename)
        img.save(out_path, quality=95)
        print(f"Generated reference image: {out_path}")

if __name__ == "__main__":
    generate_all()
