import cv2
import numpy as np
from preprocessing.preprocess_leaf import preprocess_leaf
from segmentation.auto_threshold import auto_segment_lesions
import matplotlib.pyplot as plt

# -------- CONFIG --------
image_path = "data/medium/medium2.JPG"  # Try other images too
debug = True

# -------- Load Image --------
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# -------- Step 1–2: Preprocess --------
gray_filtered, leaf_mask = preprocess_leaf(image_path, debug=debug)
if gray_filtered is None or leaf_mask is None:
    raise ValueError("Preprocessing failed, check input or function.")

# Debug: Display preprocessed images
if debug:
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grayscale Filtered")
    plt.imshow(gray_filtered, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Leaf Mask")
    plt.imshow(leaf_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# -------- Step 3: Segment --------
lesion_mask, T = auto_segment_lesions(
    gray_filtered=gray_filtered,
    leaf_mask=leaf_mask,
    image_bgr=image_bgr,
    debug=debug
)
if lesion_mask is None:
    raise ValueError("Segmentation failed, check parameters or function.")

# Debug: Display final mask
if debug:
    plt.figure(figsize=(8, 8))
    plt.title("Final Lesion Mask")
    plt.imshow(lesion_mask, cmap='gray')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# -------- Final Output --------
print(f"[INFO] Unique values in lesion_mask: {np.unique(lesion_mask)}")
print(f"[INFO] Threshold value used: {T}")

# Ensure visibility if mask is empty
lesion_mask_vis = lesion_mask.copy()
if np.count_nonzero(lesion_mask_vis) == 0:
    print("[WARN] Lesion mask is empty, marking fallback pixel.")
    lesion_mask_vis[0, 0] = 255

cv2.imshow("Final Lesion Mask", lesion_mask_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()