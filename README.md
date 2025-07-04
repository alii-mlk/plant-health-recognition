# 🍃 Leaf Infection Severity Detection & Spread Simulation

This project performs **automatic detection**, **feature-based severity estimation**, and **infection spread simulation** on leaf images. It combines traditional computer vision techniques with a trained regression model to assess and visualize plant health.

---

## 📁 Project Structure & Pipeline

The system runs in 7 structured steps:

###  1. Preprocessing
`preprocessing/preprocess_leaf.py`
- Isolates the leaf region from the image background
- Converts to grayscale, smooths the image

Test file: `test_preprocess_leaf.py`

---

###  2. Segmentation
`segmentation/auto_threshold.py`
- Uses adaptive, Otsu, and color thresholding to segment lesion areas

Test file: `test_auto_threshold.py`

---

###  3. Feature Extraction
`features/extract_features.py`
- Extracts 11 features (color, texture, shape) from lesion region
- Used for severity prediction

Test file: `test_extract_features.py`

---

###  4. Dataset Creation
`build_dataset.py`
- Runs the pipeline on a dataset of leaf images
- Extracts features and computes severity scores using fixed coefficients (`beta`)
- Saves results to `leaf_dataset_features.csv`

---

###  5. Model Training
`train_model.py`
- Trains a **linear regression model** using `scikit-learn`
- Saves model to `models/leaf_severity_model.pkl`

---

###  6. Severity Prediction for New Images
`main.py`
- Runs the full pipeline on a new image
- Predicts severity using the trained model
- Displays and logs the result

---

###  7. Infection Spread Simulation
`simulate_infection_spread.py`
- Reads an image from `prediction/input/`
- Detects lesions and simulates their spread over multiple steps
- At each step:
  - Extracts features
  - Predicts severity
  - Saves two GIFs to `prediction/output/`:
    - A **heatmap-based** infection spread GIF
    - A **real-color overlay** GIF showing lesion texture growing naturally
- Also saves each intermediate lesion mask and overlay as individual `.png` images

If no lesion is detected, the system logs it and exits gracefully without generating output.

---

## 🛠 Installation

###  Requirements

Make sure you have Python version ≥ 3.12

Install required packages:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
numpy
pandas
opencv-python
imageio
joblib
scikit-learn
```

---

## ▶️ How to Run

###  To build dataset:
```bash
python build_dataset.py
```

###  To train model:
```bash
python train_model.py
```

###  To predict severity of new image:
Place the image in `prediction/input/` and run:
```bash
python main.py
```

###  To simulate infection spread:
Make sure model and image are ready in `prediction/input/`, then run:
```bash
python simulate_infection_spread.py
```

---

## Output

- `leaf_dataset_features.csv`: All extracted features + severity scores
- `models/leaf_severity_model.pkl`: Trained linear regression model
- `prediction/output/`: Folder with:
  - `spread_step_*.png`: Binary mask of lesions at each step
  - `spread_overlay_*.png`: Lesion spread visualized over real leaf
  - `infection_spread.gif`: Animated heatmap
  - `infection_spread_on_leaf.gif`: Animated real overlay

---
