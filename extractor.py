import numpy as np

from features import *

import cv2

def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    return img

def get_features_from_path(path: str) -> np.ndarray:
    # Load image
    img = cv2.imread(path)
    
    # Segment hand
    hand, mask = sgm(img)

    # Preprocess image - includes resizing to small enough size for fast extraction
    hand = preprocess(hand)

    # Extract features
    features = hog_feature_opencv(hand)

    return features
