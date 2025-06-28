import cv2
import matplotlib.pyplot as plt
from preprocessing.preprocess_leaf import preprocess_leaf
from segmentation.auto_threshold import auto_segment_lesions
from features.extract_features import extract_features

image_path = "data/serious/serious3.JPG"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError("Image not found.")

# Preprocess
gray_filtered, leaf_mask = preprocess_leaf(image_path, debug=False)

# Lesion segmentation
lesion_mask, threshold_value = auto_segment_lesions(gray_filtered, leaf_mask, image_bgr, debug=True)

# Feature extraction
features = extract_features(image_bgr, gray_filtered, lesion_mask, leaf_mask)

# Show all images
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original")
axs[0, 0].axis('off')

axs[0, 1].imshow(leaf_mask, cmap='gray')
axs[0, 1].set_title("Leaf Mask")
axs[0, 1].axis('off')

axs[1, 0].imshow(gray_filtered, cmap='gray')
axs[1, 0].set_title("Grayscale")
axs[1, 0].axis('off')

axs[1, 1].imshow(lesion_mask, cmap='gray')
axs[1, 1].set_title("Lesion Mask")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nImage: {image_path}")
print(f"Threshold used: {threshold_value:.1f}")
print("Extracted features:")
for key, value in features.items():
    print(f" - {key}: {value:.4f}")
