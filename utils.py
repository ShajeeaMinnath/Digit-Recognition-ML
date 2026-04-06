import numpy as np
import cv2
from skimage.feature import hog

def preprocess_image(img_array):
    # 1. Resize to MNIST size
    img_resized = cv2.resize(img_array, (28, 28))
    
    # 2. Normalize raw pixels (784 features)
    raw_pixels = img_resized.flatten() / 255.0
    
    # 3. Extract HOG (Must match training exactly!)
    hog_feats = hog(img_resized, orientations=9, 
                    pixels_per_cell=(7, 7), 
                    cells_per_block=(2, 2))
    
    # 4. Combine
    return np.hstack([raw_pixels, hog_feats]).reshape(1, -1)
