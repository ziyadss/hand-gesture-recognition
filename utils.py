import numpy as np
from skimage import color, filters, io, measure, morphology

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

    # Apply a threshold
    threshold = filters.threshold_otsu(grayscale)
    binary = grayscale > threshold

    # Remove small objects
    cleaned = morphology.remove_small_objects(binary, min_size=64, connectivity=2)

    # Fill holes
    filled = morphology.remove_small_holes(cleaned, area_threshold=64, connectivity=2)

    # Label regions
    labels = morphology.label(filled)
    regions = measure.regionprops(labels)

    # Find the largest region
    largest_region = max(regions, key=lambda region: region.area)
    label = largest_region.label

    # Extract the hand
    hand = np.zeros_like(grayscale)
    hand[labels == label] = grayscale[labels == label]

    binary_hand = ~(hand > 0)

    hand_region = measure.regionprops(binary_hand.astype(int))[0]

    hull = morphology.convex_hull_image(binary_hand)

    return binary_hand, hand_region, hull


def get_features(region) -> dict:
    features = {}
    for key in region:
        if key in SKIP:
            continue
        features[key] = region[key]
    return features


def get_features_from_path(path: str) -> dict:
    binary_hand, hand_region, hull = segment_hand(path)

    return get_features(hand_region)
