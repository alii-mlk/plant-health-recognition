# main.py

import cv2
import joblib
import pandas as pd
from preprocessing.preprocess_leaf import preprocess_leaf
from segmentation.auto_threshold import auto_segment_lesions
from features.extract_features import extract_features

# -------- CONFIG --------
image_path = "prediction/input/unseen_leaf3.jpg"  # <-- put your test image here
model_path = "models/leaf_severity_model.pkl"

print("[INFO] Loading input image...")
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# -------- PIPELINE --------

# Step 1: Preprocess (leaf isolation and smoothing)
print("[INFO] Preprocessing...")
gray_filtered, leaf_mask = preprocess_leaf(image_path, debug=False)

# Step 2: Lesion segmentation (auto thresholding)
print("[INFO] Segmenting lesion areas...")
lesion_mask, _ = auto_segment_lesions(gray_filtered, leaf_mask, image_bgr=image_bgr)

# Step 3: Feature extraction
print("[INFO] Extracting features...")
features = extract_features(image_bgr, gray_filtered, lesion_mask, leaf_mask)
input_df = pd.DataFrame([features])

# Step 4: Load trained regression model
print("[INFO] Loading model...")
model = joblib.load(model_path)

# Step 5: Predict severity
print("[INFO] Predicting severity...")
predicted_severity = model.predict(input_df)[0]
print(f"Predicted Severity Score: {predicted_severity:.2f}")

# Step 6 (optional): Category
if predicted_severity < 25:
    category = "Normal"
elif predicted_severity < 50:
    category = "Minor"
elif predicted_severity < 75:
    category = "Medium"
else:
    category = "Serious"

print(f"Severity Category: {category}")
