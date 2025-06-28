import cv2
import numpy as np

def preprocess_leaf(image_path, debug=False):
    """
    Step 1 & 2 from the paper:
    - Isolate full leaf using HSV masking (green/yellow/brown)
    - Convert to grayscale and denoise with median filter
    Returns:
        - gray_filtered: grayscale preprocessed image
        - leaf_mask: binary mask where 255 = leaf, 0 = background
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"[Preprocessing] Could not read image at: {image_path}")

    if debug:
        cv2.imshow("Original", image)

    # Convert to HSV and apply broad leaf mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_leaf = np.array([15, 30, 30])
    upper_leaf = np.array([100, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

    # Mask image
    leaf_only = cv2.bitwise_and(image, image, mask=leaf_mask)

    # Grayscale + smoothing
    gray = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.medianBlur(gray, 5)

    if debug:
        cv2.imshow("Leaf Mask", leaf_mask)
        cv2.imshow("Filtered Grayscale", gray_filtered)

    return gray_filtered, leaf_mask
