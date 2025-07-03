import os
import cv2
import numpy as np
import pandas as pd
import joblib
import imageio
from preprocessing.preprocess_leaf import preprocess_leaf
from segmentation.auto_threshold import auto_segment_lesions
from features.extract_features import extract_features

def simulate_infection_spread(lesion_mask, leaf_mask, steps=10, infection_prob=0.7):
    lesion = lesion_mask.copy()
    lesion = cv2.threshold(lesion, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

    results = [lesion.copy()]
    for step in range(steps):
        dilated = cv2.dilate(lesion, kernel, iterations=1)
        spread_zone = ((dilated == 255) & (lesion == 0) & (leaf_mask > 0))
        rand_mask = (np.random.rand(*lesion.shape) < infection_prob)
        new_infection = spread_zone & rand_mask
        lesion[new_infection] = 255
        results.append(lesion.copy())
    return results

if __name__ == "__main__":
    input_path = "prediction/input/unseen_leaf4.jpg"
    output_dir = "prediction/output"
    overlay_dir = os.path.join(output_dir, "overlay")
    gif_path = os.path.join(output_dir, "infection_spread.gif")
    gif_overlay_path = os.path.join(output_dir, "infection_spread_on_leaf.gif")
    model_path = "models/leaf_severity_model.pkl"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Step 1: Load image
    print("[INFO] Loading image and model...")
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    model = joblib.load(model_path)

    # Step 2: Preprocessing
    gray_filtered, leaf_mask = preprocess_leaf(input_path, debug=False)
    lesion_mask, _ = auto_segment_lesions(gray_filtered, leaf_mask, image_bgr=image_bgr)

    if np.count_nonzero(lesion_mask) < 10:
        print("[INFO] No lesion detected — severity too low to simulate infection spread.")
        # Optionally show severity anyway
        features = extract_features(image_bgr, gray_filtered, lesion_mask, leaf_mask)
        features_df = pd.DataFrame([features])
        severity_score = model.predict(features_df)[0]
        print(f"[INFO] Severity Score: {severity_score:.2f}")

        # Save one frame with text overlay
        result = image_bgr.copy()
        cv2.putText(result, "No lesion detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result, f"Severity Score: {severity_score:.2f}", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        fallback_path = os.path.join("prediction/output", "no_lesion.png")
        cv2.imwrite(fallback_path, result)
        print(f"[INFO] Saved fallback image to {fallback_path}")
        exit()


    # Step 3: Simulate infection progression
    print("[INFO] Simulating infection spread...")
    spread_steps = simulate_infection_spread(lesion_mask, leaf_mask, steps=10, infection_prob=0.7)

    mask_gif_frames = []
    overlay_gif_frames = []

    for i, lesion_step_mask in enumerate(spread_steps):
        # Step 4: Predict severity for each step
        features = extract_features(image_bgr, gray_filtered, lesion_step_mask, leaf_mask)
        features_df = pd.DataFrame([features])
        severity_score = model.predict(features_df)[0]

        # Step 5: Create heatmap + label
        color_mask = cv2.applyColorMap(lesion_step_mask, cv2.COLORMAP_HOT)
        labeled = color_mask.copy()
        cv2.putText(labeled, f"Step {i} | Severity: {severity_score:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save heatmap image and for GIF
        cv2.imwrite(os.path.join(output_dir, f"spread_step_{i}.png"), labeled)
        mask_gif_frames.append(labeled)
        cv2.imshow(f"Spread Step {i}", labeled)
        cv2.waitKey(500)
        print(f"[INFO] Step {i}: Severity Score = {severity_score:.2f}")

           # Step 6: Grow lesion using original texture
        if i == 0:
            # Extract original lesion texture from image using step 0 mask
            lesion_base_texture = image_bgr.copy()
            lesion_base_texture[lesion_step_mask == 0] = 0  # keep only lesion part
            overlay_gif_frames.append(image_bgr.copy())  # add original as first frame
            # Also save original
            overlay_path = os.path.join(overlay_dir, f"spread_overlay_{i}.png")
            cv2.imwrite(overlay_path, image_bgr)
            continue

        # Find newly infected pixels between steps
        previous_mask = spread_steps[i - 1]
        current_mask = lesion_step_mask
        new_growth = (current_mask == 255) & (previous_mask == 0)

        # Start with the previous blended image so infection is cumulative
        blended = overlay_gif_frames[-1].copy()

        # Apply real lesion color (from step 0) to new growth pixels
        blended[new_growth] = lesion_base_texture[new_growth]

        # Save this new overlay frame
        overlay_path = os.path.join(overlay_dir, f"spread_overlay_{i}.png")
        cv2.imwrite(overlay_path, blended)
        overlay_gif_frames.append(blended)
    # Save GIF from heatmaps
    print("[INFO] Saving mask-based GIF...")
    imageio.mimsave(gif_path, mask_gif_frames, duration=2)

    # Save GIF from overlays
    print("[INFO] Saving overlay-on-leaf GIF...")
    imageio.mimsave(gif_overlay_path, overlay_gif_frames, duration=2)

    print(f"[INFO] GIF saved: {gif_path}")
    print(f"[INFO] Overlay GIF saved: {gif_overlay_path}")
    print("[INFO] Done. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
