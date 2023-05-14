import cv2
import numpy as np

from features import *


def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img


def get_features_from_path(path: str) -> np.ndarray:
    # Load image
    img = cv2.imread(path)

    # Segment hand
    mask = sgm(img)
    hand = cv2.bitwise_and(img, img, mask=mask)

    # Preprocess image - includes resizing to small enough size for fast extraction
    hand = preprocess(hand)

    # Extract features
    hog = hog_feature_opencv(hand)
    lbp = get_9ulbp_features(hand, (8, 8))

    features = np.concatenate((hog, lbp))

    return features
