from preprocessing.preprocess_leaf import preprocess_leaf
import cv2

image_path = "data/serious/9.JPG"
debug = True

filtered = preprocess_leaf(image_path, debug=debug)

cv2.waitKey(0)
cv2.destroyAllWindows()
