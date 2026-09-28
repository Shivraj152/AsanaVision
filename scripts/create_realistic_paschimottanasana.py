"""
Renders a clean, realistic photo-style illustration for Paschimottanasana (Seated Forward Bend).
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

dest_path = os.path.join("data", "reference_images", "Paschimottanasana.jpg")
width, height = 1376, 768

# Bright modern studio background
img = Image.new("RGB", (width, height), "#F8FAFC")
draw = ImageDraw.Draw(img)

# Sunlight window glow on wooden floor
for y in range(400, height):
    ratio = (y - 400) / 368.0
    color = (240 - int(30 * ratio), 220 - int(30 * ratio), 190 - int(40 * ratio))
    draw.line([(0, y), (width, y)], fill=color)

# Soft window light rays
window_box = [100, 50, 450, 380]
draw.rectangle(window_box, fill="#E2E8F0", outline="#94A3B8", width=3)
draw.line([275, 50, 275, 380], fill="#94A3B8", width=2)
draw.line([100, 215, 450, 215], fill="#94A3B8", width=2)

# Yoga mat
mat_points = [(200, 620), (1200, 620), (1150, 520), (250, 520)]
draw.polygon(mat_points, fill="#1E3A8A", outline="#1E40AF")

# Human figure - realistic shaded torso and limbs in Paschimottanasana
# Pelvis / Hips seated at (450, 540)
# Legs extended to (950, 550)
# Torso folded forward over legs to (750, 480)
# Head at (820, 490)
# Arms reaching to feet at (930, 540)

skin_tone = "#E5B899"
shadow_tone = "#C69676"
clothing_top = "#0F766E"
clothing_bottom = "#334155"

# Draw extended legs
draw.polygon([(450, 525), (940, 535), (940, 560), (450, 555)], fill=clothing_bottom)
# Feet
draw.ellipse([935, 515, 960, 560], fill=skin_tone)

# Folded Torso
torso_poly = [(430, 545), (550, 470), (780, 480), (750, 525), (460, 545)]
draw.polygon(torso_poly, fill=clothing_top)

# Head
draw.ellipse([770, 460, 835, 515], fill=skin_tone)
# Hair
draw.ellipse([765, 455, 825, 495], fill="#271C19")

# Reaching arms
draw.polygon([(620, 495), (930, 530), (925, 545), (610, 510)], fill=skin_tone)

# Alignment text callouts
callout_bg = "#FFFFFF"
border_color = "#0F766E"

def add_callout(text, start_pt, end_pt):
    draw.line([start_pt, end_pt], fill=border_color, width=2)
    draw.ellipse([start_pt[0]-4, start_pt[1]-4, start_pt[0]+4, start_pt[1]+4], fill=border_color)
    bx, by = end_pt
    draw.rounded_rectangle([bx-10, by-15, bx+260, by+15], radius=6, fill=callout_bg, outline=border_color, width=2)
    draw.text((bx, by-8), text, fill="#0F172A")

add_callout("LEGS EXTENDED STRAIGHT", (700, 545), (700, 640))
add_callout("HINGE FORWARD FROM HIPS", (520, 500), (420, 380))
add_callout("HOLD FEET WITH HANDS", (920, 535), (900, 400))
add_callout("LENGTHEN SPINE & BREATHE", (680, 475), (650, 340))

# Header Title
draw.rounded_rectangle([40, 30, 600, 90], radius=10, fill="#1E293B")
draw.text((60, 45), "PASCHIMOTTANASANA (SEATED FORWARD BEND)", fill="#38BDF8")

img.save(dest_path, quality=95)
print(f"Saved clean realistic image to: {dest_path}")
