import os
import csv
import cv2
from preprocessing.preprocess_leaf import preprocess_leaf
from segmentation.auto_threshold import auto_segment_lesions
from features.extract_features import extract_features

# Severity weight coefficients
beta = {
    'Hue': 0.049, 'Saturation': 0.022, 'Value': 0.017,
    'Energy': 1.818, 'Homogeneity': 1.786, 'Smoothness': 1.000,
    '3rd Moment': 0.000, 'Consistency': 0.049, 'Entropy': 0.485,
    'Degree of Rectification': 1.000, 'Density': 50.000
}

def calculate_severity_score(features):
    return sum(beta[k] * features[k] for k in beta)

severity_ranges = {
    'normal': (0, 25), 'minor': (25, 50),
    'medium': (50, 75), 'serious': (75, 100)
}

def scale_severity_score(score, image_type):
    min_val, max_val = severity_ranges[image_type]
    min_score, max_score = 0, 100  # Placeholder
    return min_val + (max_val - min_val) * (score - min_score) / (max_score - min_score)

DATA_DIR = "data"
OUTPUT_CSV = "leaf_dataset_features.csv"

header = [
    "Filename", "Hue", "Saturation", "Value",
    "Energy", "Homogeneity", "Smoothness",
    "3rd Moment", "Consistency", "Entropy",
    "Degree of Rectification", "Density", "Severity"
]
rows = []

for subdir, _, files in os.walk(DATA_DIR):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        rel_path = os.path.join(subdir, file)
        print(f"[INFO] Processing {rel_path}")

        image_bgr = cv2.imread(rel_path)
        if image_bgr is None:
            print(f"[ERROR] Couldn't load image: {rel_path}")
            continue

        try:
            gray_filtered, leaf_mask = preprocess_leaf(rel_path, debug=False)
            lesion_mask, _ = auto_segment_lesions(gray_filtered, leaf_mask, image_bgr, debug=False)
            features = extract_features(image_bgr, gray_filtered, lesion_mask, leaf_mask)
            severity_score = calculate_severity_score(features)
            image_type = 'medium' if 'medium' in file.lower() else \
                         'minor' if 'minor' in file.lower() else \
                         'serious' if 'serious' in file.lower() else 'normal'
            scaled_score = scale_severity_score(severity_score, image_type)

            feature_values = [
                file, features["Hue"], features["Saturation"], features["Value"],
                features["Energy"], features["Homogeneity"], features["Smoothness"],
                features["3rd Moment"], features["Consistency"], features["Entropy"],
                features["Degree of Rectification"], features["Density"], scaled_score
            ]
            rows.append(feature_values)

        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")
            continue

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Dataset built with {len(rows)} samples → {OUTPUT_CSV}")
