"""
Reference Guide & Visual Diagrams for AsanaVision.
Provides step-by-step execution instructions, target angle reference cards, Sanskrit typography, difficulty levels, and visual pose illustrations for 20 yoga poses.
"""

import os
from typing import Dict, Any

YOGA_POSE_GUIDES: Dict[str, Dict[str, Any]] = {
    "Tadasana": {
        "english_name": "Mountain Pose",
        "sanskrit_script": "ताडासन",
        "difficulty": "Beginner",
        "category": "Standing Baseline",
        "image_file": "data/reference_images/Tadasana.jpg",
        "instructions": [
            "Stand straight with feet together or hip-width apart.",
            "Distribute body weight evenly across both feet.",
            "Keep knees unbent, engage thighs, and pull up knee caps.",
            "Keep your spine tall, shoulders relaxed and level, and arms extended down."
        ],
        "target_angles_summary": "Knees: 155°–180° | Torso: Vertical (0°–18°) | Shoulders: Level (0°–15°)"
    },
    "Bhujangasana": {
        "english_name": "Cobra Pose",
        "sanskrit_script": "भुजंगासन",
        "difficulty": "Beginner",
        "category": "Backbend / Prone",
        "image_file": "data/reference_images/Bhujangasana.jpg",
        "instructions": [
            "Lie flat on your stomach with tops of feet on the mat.",
            "Place palms under your shoulders with elbows bent close to your torso.",
            "Inhale and lift your chest off the floor by arching your upper back.",
            "Press feet into mat, keep shoulders relaxed down away from ears."
        ],
        "target_angles_summary": "Elbows: Extended (120°–180°) | Torso: Arched Up (25°–75°) | Legs: Extended (150°–180°)"
    },
    "Vajrasana": {
        "english_name": "Thunderbolt Pose",
        "sanskrit_script": "वज्रासन",
        "difficulty": "Beginner",
        "category": "Seated Kneeling",
        "image_file": "data/reference_images/Vajrasana.jpg",
        "instructions": [
            "Kneel on the floor with knees close together.",
            "Sit back onto your heels so your buttocks rest on your soles.",
            "Keep your spine straight, torso upright, and shoulders relaxed.",
            "Place hands flat on your thighs."
        ],
        "target_angles_summary": "Knees: Folded (< 50°) | Torso: Vertical (0°–20°) | Shoulders: Level (0°–15°)"
    },
    "Anantasana": {
        "english_name": "Side-Reclining Leg Lift",
        "sanskrit_script": "अनन्तासन",
        "difficulty": "Intermediate",
        "category": "Reclining / Side Stretch",
        "image_file": "data/reference_images/Anantasana.jpg",
        "instructions": [
            "Lie on your side with lower arm supporting head.",
            "Bend top knee and grasp big toe with fingers.",
            "Extend top leg straight up towards ceiling.",
            "Keep bottom leg firmly grounded and body balanced."
        ],
        "target_angles_summary": "Raised Leg Knee: Straight (> 140°) | Lifted Hip Angle: High (> 100°)"
    },
    "Ardhakati Chakrasana": {
        "english_name": "Standing Side Bend",
        "sanskrit_script": "अर्धकटिचक्रासन",
        "difficulty": "Beginner",
        "category": "Standing Lateral Bend",
        "image_file": "data/reference_images/Ardhakati_Chakrasana.jpg",
        "instructions": [
            "Stand upright in Tadasana with feet together.",
            "Raise one arm straight overhead touching your ear.",
            "Inhale deeply, then exhale while bending sideways to the opposite side.",
            "Keep legs straight and weight equal on both feet."
        ],
        "target_angles_summary": "Raised Arm: Extended (> 140°) | Torso Tilt: Side Bend (15°–50°) | Knees: Straight (> 150°)"
    },
    "Kati Chakrasana": {
        "english_name": "Standing Spinal Twist",
        "sanskrit_script": "कटिचक्रासन",
        "difficulty": "Beginner",
        "category": "Standing Twist",
        "image_file": "data/reference_images/Kati_Chakrasana.jpg",
        "instructions": [
            "Stand with feet shoulder-width apart.",
            "Extend arms out horizontally at shoulder level.",
            "Twist upper torso to the side, wrapping arms around waist.",
            "Look back over your shoulder while keeping hips square."
        ],
        "target_angles_summary": "Shoulder Twist: Rotated (10°–60°) | Legs: Stable & Straight (> 150°)"
    },
    "Marjariasana": {
        "english_name": "Cat-Cow Pose",
        "sanskrit_script": "मार्जरीआसन",
        "difficulty": "Beginner",
        "category": "Spinal Mobility / Quadruped",
        "image_file": "data/reference_images/Marjariasana.jpg",
        "instructions": [
            "Start on hands and knees in tabletop position.",
            "Align wrists directly under shoulders and knees under hips.",
            "Inhale: arch your back downwards, lift tailbone and chest (Cow).",
            "Exhale: round your spine upwards, tuck chin to chest (Cat)."
        ],
        "target_angles_summary": "Elbows: Straight (145°–180°) | Knees: 90° under Hips (75°–110°)"
    },
    "Parvatasana": {
        "english_name": "Downward Mountain / V-Shape",
        "sanskrit_script": "पर्वतासन",
        "difficulty": "Beginner",
        "category": "Inversion / Extension",
        "image_file": "data/reference_images/Parvatasana.jpg",
        "instructions": [
            "From hands and knees, lift knees and press hips up towards ceiling.",
            "Press palms firmly into mat with arms fully extended.",
            "Form an inverted 'V' shape with body.",
            "Keep heels pressing towards floor and spine lengthened."
        ],
        "target_angles_summary": "Arms: Extended (> 150°) | Hips Angle: Inverted V (60°–120°)"
    },
    "Sarvangasana": {
        "english_name": "Shoulderstand",
        "sanskrit_script": "सर्वांगासन",
        "difficulty": "Advanced",
        "category": "Inverted Balance",
        "image_file": "data/reference_images/Sarvangasana.jpg",
        "instructions": [
            "Lie on back, lift legs and hips overhead into Shoulderstand.",
            "Support lower back with palms, elbows grounded on mat.",
            "Extend legs and torso straight up vertically toward ceiling.",
            "Rest weight on shoulders and upper back, not neck."
        ],
        "target_angles_summary": "Torso Verticality: Inverted Upright (140°–180°) | Knees: Straight (> 150°)"
    },
    "Viparita Karani": {
        "english_name": "Legs-Up-The-Wall Pose",
        "sanskrit_script": "विपरीतकरणी",
        "difficulty": "Beginner",
        "category": "Restorative Inversion",
        "image_file": "data/reference_images/Viparita_Karani.jpg",
        "instructions": [
            "Lie flat on your back near a wall.",
            "Extend legs straight up against wall at 90 degree angle to torso.",
            "Rest arms comfortably by sides with palms facing up.",
            "Keep back flat and hips grounded on floor."
        ],
        "target_angles_summary": "Knees: Straight (> 150°) | Hip-Torso Angle: 90° L-shape (70°–115°)"
    },
    "Vrikshasana": {
        "english_name": "Tree Pose",
        "sanskrit_script": "वृक्षासन",
        "difficulty": "Beginner",
        "category": "Standing Balance",
        "image_file": "data/reference_images/Vrikshasana.jpg",
        "instructions": [
            "Stand tall in Tadasana and shift weight onto your left leg.",
            "Place sole of right foot against inner left thigh or calf (avoid knee).",
            "Bring palms together in front of chest (Anjali Mudra) or extend arms overhead.",
            "Focus your gaze on a steady point in front of you."
        ],
        "target_angles_summary": "Standing Leg Knee: Straight (155°–180°) | Bent Knee: Folded (35°–85°) | Torso: Upright (0°–15°)"
    },
    "Trikonasana": {
        "english_name": "Triangle Pose",
        "sanskrit_script": "त्रिकोणासन",
        "difficulty": "Beginner",
        "category": "Standing Lateral Stretch",
        "image_file": "data/reference_images/Trikonasana.jpg",
        "instructions": [
            "Stand with feet wide apart (3-4 feet). Turn right foot out 90 degrees.",
            "Extend arms sideways parallel to the floor at shoulder level.",
            "Hinge sideways at right hip and bring right hand down to shin or floor.",
            "Extend left arm straight up toward ceiling, looking up at left fingertips."
        ],
        "target_angles_summary": "Both Knees: Straight (> 150°) | Arms Angle: Extended T-Shape (150°–180°) | Torso Tilt: Lateral (35°–70°)"
    },
    "Virabhadrasana I": {
        "english_name": "Warrior I Pose",
        "sanskrit_script": "वीरभद्रासन १",
        "difficulty": "Intermediate",
        "category": "Standing Lunge",
        "image_file": "data/reference_images/Virabhadrasana_I.jpg",
        "instructions": [
            "Step left foot back 3-4 feet, angling foot outward at 45 degrees.",
            "Bend right front knee to a 90 degree angle directly over right ankle.",
            "Square chest and hips forward towards front of mat.",
            "Reach both arms straight overhead parallel with palms facing each other."
        ],
        "target_angles_summary": "Front Knee: Bent (80°–110°) | Back Knee: Extended (150°–180°) | Arms: Raised Overhead (> 145°)"
    },
    "Virabhadrasana II": {
        "english_name": "Warrior II Pose",
        "sanskrit_script": "वीरभद्रासन २",
        "difficulty": "Intermediate",
        "category": "Standing Wide Lunge",
        "image_file": "data/reference_images/Virabhadrasana_II.jpg",
        "instructions": [
            "Step feet wide apart. Turn right foot out 90 degrees and left foot slightly in.",
            "Bend right knee to 90 degrees directly above right ankle.",
            "Extend arms out parallel to floor, reach actively toward opposite walls.",
            "Gaze forward over front right fingertips with torso upright."
        ],
        "target_angles_summary": "Front Knee: Bent 90° (75°–105°) | Back Knee: Straight (> 150°) | Arms: Horizontal T-Shape (160°–180°)"
    },
    "Utkatasana": {
        "english_name": "Chair Pose",
        "sanskrit_script": "उत्कटासन",
        "difficulty": "Intermediate",
        "category": "Standing Power Squat",
        "image_file": "data/reference_images/Utkatasana.jpg",
        "instructions": [
            "Stand with feet together or hip-width apart.",
            "Inhale and raise arms straight overhead alongside ears.",
            "Exhale and bend knees, sitting hips back as if sitting in a chair.",
            "Keep weight in heels and chest lifted."
        ],
        "target_angles_summary": "Knees: Bent Squat (80°–130°) | Arms: Extended Overhead (> 140°) | Torso: Slanted (20°–50°)"
    },
    "Setu Bandhasana": {
        "english_name": "Bridge Pose",
        "sanskrit_script": "सेतुबंधासन",
        "difficulty": "Intermediate",
        "category": "Supine Backbend",
        "image_file": "data/reference_images/Setu_Bandhasana.jpg",
        "instructions": [
            "Lie on your back with knees bent and feet flat on floor hip-width apart.",
            "Extend arms along floor with palms facing down near heels.",
            "Press feet and arms into floor, inhale and lift hips up toward ceiling.",
            "Roll shoulders underneath body and interlace fingers if comfortable."
        ],
        "target_angles_summary": "Hips: Lifted Arch (130°–175°) | Knees: Bent (70°–110°) | Torso: Elevated Bridge"
    },
    "Balasana": {
        "english_name": "Child's Pose",
        "sanskrit_script": "बालासन",
        "difficulty": "Beginner",
        "category": "Restorative Kneeling Fold",
        "image_file": "data/reference_images/Balasana.jpg",
        "instructions": [
            "Kneel on floor, touch big toes together and sit on heels.",
            "Separate knees hip-width apart and fold torso forward between thighs.",
            "Rest forehead gently on mat and extend arms forward palms down.",
            "Relax shoulders, breathe deeply and release spine tension."
        ],
        "target_angles_summary": "Knees: Fully Folded (< 45°) | Hips: Flexed Fold (< 45°) | Arms: Extended Forward"
    },
    "Paschimottanasana": {
        "english_name": "Seated Forward Bend",
        "sanskrit_script": "पश्चिमोत्तानासन",
        "difficulty": "Intermediate",
        "category": "Seated Fold",
        "image_file": "data/reference_images/Paschimottanasana.jpg",
        "instructions": [
            "Sit upright on floor with legs fully extended together in front.",
            "Inhale, lengthen spine and reach arms overhead.",
            "Exhale and hinge from hip joints, folding torso over thighs.",
            "Hold feet, ankles, or shins with hands while keeping knees straight."
        ],
        "target_angles_summary": "Knees: Fully Extended (> 150°) | Hip-Torso Angle: Deep Fold (< 60°)"
    },
    "Ustrasana": {
        "english_name": "Camel Pose",
        "sanskrit_script": "उष्ट्रासन",
        "difficulty": "Intermediate",
        "category": "Kneeling Backbend",
        "image_file": "data/reference_images/Ustrasana.jpg",
        "instructions": [
            "Kneel upright with knees hip-width apart and thighs perpendicular to floor.",
            "Place palms on lower back with fingers pointing down.",
            "Inhale, lift chest and lean back, reaching hands down to grasp heels.",
            "Keep hips over knees and arch upper spine backwards comfortably."
        ],
        "target_angles_summary": "Torso Backbend: Arched (30°–80°) | Knees: Kneeling 90° (75°–110°) | Arms: Reach Heels"
    },
    "Halasana": {
        "english_name": "Plow Pose",
        "sanskrit_script": "हलासन",
        "difficulty": "Advanced",
        "category": "Inverted Forward Fold",
        "image_file": "data/reference_images/Halasana.jpg",
        "instructions": [
            "Lie on back in Sarvangasana (Shoulderstand) position.",
            "Lower legs overhead behind head until toes touch floor behind you.",
            "Keep legs straight and knees unbent.",
            "Interlace fingers behind back on mat or support lower back."
        ],
        "target_angles_summary": "Torso-Hip Inversion: Folded Overhead (40°–90°) | Knees: Fully Extended (> 145°)"
    }
}
