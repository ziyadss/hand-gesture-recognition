import cv2
import numpy as np
from skimage import (
    color,
    exposure,
    feature,
    filters,
    io,
    measure,
    morphology,
    transform,
)

lower = np.array([0, 20, 70])
upper = np.array([20, 255, 255])
kernel = np.ones((5, 5), np.uint8)


def sgm(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    hand = cv2.bitwise_and(img, img, mask=mask)
    return hand, mask


def segment_no_shadows(img: np.ndarray) -> np.ndarray:
    hand, mask = sgm(img)
    pixel_values = hand.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, flags
    )

    labels = labels.reshape(hand.shape[:2])

    non_black_labels = labels[mask > 0]
    unique, counts = np.unique(non_black_labels, return_counts=True)
    hand_label = unique[np.argmax(counts)]

    cluster_mask = np.int8(labels == hand_label)
    cluster = cv2.bitwise_and(img, img, mask=cluster_mask)  # type: ignore

    return cluster


winSize = (64, 64)
blockSize = (64, 64)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


def hog_feature_opencv(gray):
    return hog.compute(gray).ravel()


def lbp_feature_opencv(gray):
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp = cv2.normalize(lbp, lbp, 0, 255, cv2.NORM_MINMAX)
    return lbp.ravel()


def get_features_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    hand = segment_no_shadows(img)
    resized = cv2.resize(hand, None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_AREA)  # type: ignore
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hog = hog_feature_opencv(gray)
    lbp = lbp_feature_opencv(gray)

    return np.concatenate((hog, lbp))


SKIP = {
    "coords",
    "image",
    "image_convex",
    "image_filled",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "centroid_local",
    "slice",
    "label",
    "perimenter_crofton",
    "euler_number",
    "feret_diameter_max",
}


def segment_hand(path: str):
    image = io.imread(path)
    grayscale = color.rgb2gray(image)

    threshold = filters.threshold_otsu(grayscale)
    binary = grayscale > threshold

    cleaned = morphology.remove_small_objects(binary, min_size=64, connectivity=2)
    filled = morphology.remove_small_holes(cleaned, area_threshold=64, connectivity=2)

    labels = morphology.label(filled)
    regions = measure.regionprops(labels)

    largest_region = max(regions, key=lambda region: region.area)
    label = largest_region.label

    hand = np.zeros_like(grayscale)
    hand[labels == label] = grayscale[labels == label]

    binary_hand = ~(hand > 0)

    hand_region = measure.regionprops(binary_hand.astype(int))[0]

    hull = morphology.convex_hull_image(binary_hand)

    return binary_hand, hand_region, hull


def get_features_from_region(region) -> np.ndarray:
    features = {}
    for key in region:
        if key in SKIP:
            continue
        features[key] = region[key]

    data = []
    for entry in features:
        data.append([])
        for key in entry:
            if isinstance(entry[key], tuple):
                data[-1].extend(entry[key])
            else:
                data[-1].append(entry[key])
    return np.array(data)


def get_region_features_from_path(path: str) -> np.ndarray:
    binary_hand, hand_region, hull = segment_hand(path)

    return get_features_from_region(hand_region)


def simple_hog_feat(path: str):
    image = io.imread(path)

    grayscale = color.rgb2gray(image)

    corrected = exposure.adjust_gamma(grayscale, gamma=0.5)  # type: ignore

    resized = transform.rescale(
        corrected, 1 / 24, anti_aliasing=True, preserve_range=True
    )

    features = feature.hog(
        resized,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features
