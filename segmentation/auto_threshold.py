import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_segment_lesions(gray_filtered, leaf_mask, image_bgr=None, debug=False):
    # Step 1: Contrast Enhancement
    gray_filtered = cv2.GaussianBlur(gray_filtered, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray_filtered)
    gamma = 1.5
    gray_enhanced = np.power(gray_enhanced / 255.0, gamma) * 255
    gray_enhanced = gray_enhanced.astype(np.uint8)

    # Step 2: Otsu Thresholding
    T_otsu, binary_otsu = cv2.threshold(gray_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lesion_mask_otsu = cv2.bitwise_not(binary_otsu)
    lesion_mask_otsu = cv2.bitwise_and(lesion_mask_otsu, lesion_mask_otsu, mask=leaf_mask)

    # Step 3: Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    lesion_mask_adaptive = cv2.bitwise_and(adaptive_thresh, adaptive_thresh, mask=leaf_mask)

    # Step 4: Color-based lesion mask (brown/yellow)
    if image_bgr is not None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([0, 40, 30])
        upper_brown = np.array([30, 255, 255])
        lesion_mask_color = cv2.inRange(hsv, lower_brown, upper_brown)
        lesion_mask_color = cv2.bitwise_and(lesion_mask_color, lesion_mask_color, mask=leaf_mask)
    else:
        lesion_mask_color = np.zeros_like(leaf_mask)

    # Step 5: Greenness filtering — reduce false positives
    b, g, r = cv2.split(image_bgr)
    exg = 2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)  # Excess Green
    exg_mask = (exg > 20).astype(np.uint8) * 255  # Healthy areas
    exg_mask = cv2.bitwise_and(exg_mask, exg_mask, mask=leaf_mask)

    # Step 6: Combine masks
    lesion_mask = cv2.bitwise_or(lesion_mask_otsu, lesion_mask_adaptive)
    lesion_mask = cv2.bitwise_or(lesion_mask, lesion_mask_color)

    # Step 7: Remove pixels clearly in healthy green areas
    lesion_mask = cv2.bitwise_and(lesion_mask, cv2.bitwise_not(exg_mask))

    # Step 8: Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

    # Step 9: Neighborhood cleanup
    lesion_mask = refine_mask_with_neighborhood(lesion_mask, 3, 5)

    # Step 10: Remove small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_mask)
    sizes = stats[1:, -1]
    min_size = 60
    valid = np.isin(labels, np.where(sizes >= min_size)[0] + 1)
    lesion_mask = (valid * 255).astype(np.uint8)

    if debug:
        print(f"[DEBUG] Otsu Threshold: {T_otsu}")
        print(f"[DEBUG] Lesion pixels: {np.count_nonzero(lesion_mask)}")
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 4, 1); plt.title("Grayscale Enhanced"); plt.imshow(gray_enhanced, cmap='gray')
        plt.subplot(1, 4, 2); plt.title("Otsu"); plt.imshow(lesion_mask_otsu, cmap='gray')
        plt.subplot(1, 4, 3); plt.title("Color Mask"); plt.imshow(lesion_mask_color, cmap='gray')
        plt.subplot(1, 4, 4); plt.title("Final Lesion"); plt.imshow(lesion_mask, cmap='gray')
        plt.tight_layout(); plt.show(block=False); plt.pause(2); plt.close()

    return lesion_mask, T_otsu


def refine_mask_with_neighborhood(lesion_mask, kernel_size=3, threshold_neighbors=5):
    h, w = lesion_mask.shape
    result = lesion_mask.copy()
    for y in range(kernel_size // 2, h - kernel_size // 2):
        for x in range(kernel_size // 2, w - kernel_size // 2):
            roi = lesion_mask[y - 1:y + 2, x - 1:x + 2]
            if np.sum(roi == 255) < threshold_neighbors:
                result[y, x] = 0
    return result
