import numpy as np
from skimage import color, filters, io, morphology


def segment_hand(path: np.ndarray) -> np.ndarray:
    image = io.imread(path)

    grayscale = color.rgb2gray(image)

    # Apply a threshold
    threshold = filters.threshold_otsu(grayscale)
    binary = grayscale > threshold

    # # Remove shadow
    # shadow_mask = filters.threshold_local(grayscale, 101)
    # shadow_mask = np.uint8(grayscale > shadow_mask)
    # shadow_mask = morphology.binary_closing(shadow_mask, morphology.disk(15))
    # grayscale_no_shadow = np.where(shadow_mask, grayscale, 1)

    # # Apply a threshold
    # threshold = filters.threshold_otsu(grayscale_no_shadow)
    # binary = grayscale_no_shadow > threshold

    # Remove small objects
    cleaned = morphology.remove_small_objects(binary, min_size=64, connectivity=2)

    # Fill holes
    filled = morphology.remove_small_holes(cleaned, area_threshold=64, connectivity=2)

    # Label regions
    labels = morphology.label(filled)

    # Find the largest region
    largest_region = 0
    largest_size = 0
    for region in range(1, np.max(labels) + 1):
        size = np.sum(labels == region)
        if size > largest_size:
            largest_size = size
            largest_region = region

    # Extract the hand
    hand = np.zeros_like(grayscale)
    hand[labels == largest_region] = grayscale[labels == largest_region]

    return hand
