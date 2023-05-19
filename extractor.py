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
    features = hog_feature_opencv(hand)

    return features

def get_features_orientations(path: str) -> list[np.ndarray]:
    # Load image
    img = cv2.imread(path)

    # Segment hand
    mask = sgm(img)
    hand = cv2.bitwise_and(img, img, mask=mask)

    # Preprocess image - includes resizing to small enough size for fast extraction
    hand = preprocess(hand)

    # Get the four orientations
    orientations = get_orientations(hand)

    # Extract features
    features = [hog_feature_opencv(orientation) for orientation in orientations]

    return features

def get_orientations(img: np.ndarray) -> list[np.ndarray]:
    normal = img
    flipped_horizontally = cv2.flip(img, 1)
    flipped_vertically = cv2.flip(img, 0)
    flipped_both = cv2.flip(img, -1)
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flippedh_rotated_90 = cv2.flip(rotated_90, 1)
    flippedh_rotated_180 = cv2.flip(rotated_180, 1)
    flippedh_rotated_270 = cv2.flip(rotated_270, 1)
    flippedv_rotated_90 = cv2.flip(rotated_90, 0)
    flippedv_rotated_180 = cv2.flip(rotated_180, 0)
    flippedv_rotated_270 = cv2.flip(rotated_270, 0)
    flippedb_rotated_90 = cv2.flip(rotated_90, -1)
    flippedb_rotated_180 = cv2.flip(rotated_180, -1)
    flippedb_rotated_270 = cv2.flip(rotated_270, -1)

    return [
        normal,
        flipped_horizontally,
        flipped_vertically,
        flipped_both,
        rotated_90,
        rotated_180,
        rotated_270,
        flippedh_rotated_90,
        flippedh_rotated_180,
        flippedh_rotated_270,
        flippedv_rotated_90,
        flippedv_rotated_180,
        flippedv_rotated_270,
        flippedb_rotated_90,
        flippedb_rotated_180,
        flippedb_rotated_270,
    ]
