import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

def extract_features(image_bgr, gray_filtered, lesion_mask, leaf_mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lesion_mask_bool = lesion_mask.astype(bool)
    leaf_mask_bool = leaf_mask.astype(bool)

    if np.count_nonzero(lesion_mask_bool) == 0:
        hue = np.mean(h[leaf_mask_bool])
        sat = np.mean(s[leaf_mask_bool])
        val = np.mean(v[leaf_mask_bool])
        lesion_pixels = gray_filtered[leaf_mask_bool]
        glcm = graycomatrix(gray_filtered, [1], [0], symmetric=True, normed=True)
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        smoothness = 1 - (1 / (1 + np.var(lesion_pixels)))
        third_moment = np.mean((lesion_pixels - np.mean(lesion_pixels))**3)
        hist = cv2.calcHist([gray_filtered], [0], leaf_mask, [256], [0, 256]).flatten()
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        entropy_val = entropy(hist + 1e-6)
        consistency = np.std(lesion_pixels)
        degree_rectification = 0
        density = 0
        return {
            "Hue": hue, "Saturation": sat, "Value": val,
            "Energy": energy, "Homogeneity": homogeneity,
            "Smoothness": smoothness, "3rd Moment": third_moment,
            "Consistency": consistency, "Entropy": entropy_val,
            "Degree of Rectification": degree_rectification,
            "Density": density
        }

    hue = np.mean(h[lesion_mask_bool])
    sat = np.mean(s[lesion_mask_bool])
    val = np.mean(v[lesion_mask_bool])
    glcm = graycomatrix(gray_filtered, [1], [0], symmetric=True, normed=True)
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    lesion_pixels = gray_filtered[lesion_mask_bool]
    smoothness = 1 - (1 / (1 + np.var(lesion_pixels)))
    third_moment = np.mean((lesion_pixels - np.mean(lesion_pixels))**3)
    hist = cv2.calcHist([gray_filtered], [0], lesion_mask, [256], [0, 256]).flatten()
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    entropy_val = entropy(hist + 1e-6)
    consistency = np.std(lesion_pixels)
    x, y, w, h = cv2.boundingRect(lesion_mask)
    bounding_area = w * h
    lesion_area = np.count_nonzero(lesion_mask)
    degree_rectification = lesion_area / bounding_area if bounding_area else 0
    leaf_area = np.count_nonzero(leaf_mask)
    density = lesion_area / leaf_area if leaf_area else 0

    return {
        "Hue": hue, "Saturation": sat, "Value": val,
        "Energy": energy, "Homogeneity": homogeneity,
        "Smoothness": smoothness, "3rd Moment": third_moment,
        "Consistency": consistency, "Entropy": entropy_val,
        "Degree of Rectification": degree_rectification,
        "Density": density
    }
