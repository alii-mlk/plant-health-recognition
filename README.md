# Plant Disease Severity Detection

This project implements a full image-processing pipeline to estimate the severity of plant leaf diseases using traditional computer vision techniques and a linear regression model.

## How It Works

### 1. Preprocessing
- Isolates the leaf using HSV masking
- Converts image to grayscale
- Applies smoothing filters

### 2. Lesion Segmentation
- Uses enhanced grayscale contrast and a combination of:
  - Otsu thresholding
  - Adaptive thresholding
- Refines using the leaf mask

### 3. Feature Extraction
Extracts 11 features:
- **Color**: Hue, Saturation, Value
- **Texture**: Energy, Homogeneity, Smoothness, 3rd Moment, Consistency, Entropy
- **Shape**: Degree of Rectification, Density

### 4. Severity Scoring
- Uses a **manually weighted formula**
- Special emphasis on **lesion-to-leaf area ratio** (Density), which may be weighted more (e.g., ×50) for more realistic results

### 5. Model Training
- `train_model.py` fits a `LinearRegression` model using the features and severity scores from the CSV
- Model is saved to `models/leaf_severity_model.pkl`

### 6. Prediction
- `main.py` processes a new image end-to-end and predicts severity

---

## Input/Output

- Input images: JPEG/PNG files in `/data/`
- CSV output: `leaf_dataset_features.csv` with computed features and severity
- Trained model: `models/leaf_severity_model.pkl`

---

## Usage

1. **Build the dataset**:
```bash
python build_dataset.py
```

2. **Train the model**:
```bash
python train_model.py
```

3. **Predict on a new image**:
- Place image in `data/`
- Update filename in `main.py`
```bash
python main.py
```

---# plant-health-recognition
