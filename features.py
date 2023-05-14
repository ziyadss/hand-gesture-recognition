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


def gray_world(img):
    b, g, r = cv2.split(img)

    b_avg = np.average(b)
    g_avg = np.average(g)
    r_avg = np.average(r)
    avg = (b_avg + g_avg + r_avg) / 3

    b = np.clip(b * (avg / b_avg), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg / g_avg), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg / r_avg), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


LOWER = np.array([0, 20, 70])
UPPER = np.array([20, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)


def sgm(img):
    # p_img = gray_world(img)
    p_img = img
    hsv = cv2.cvtColor(p_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return mask


def segment_no_shadows(img: np.ndarray) -> np.ndarray:
    mask = sgm(img)
    hand = cv2.bitwise_and(img, img, mask=mask)

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


def get_lbp_code(img):
    center = img[1:-1, 1:-1]
    binary = (img[:-2, :-2] >= center) * 1
    binary |= (img[:-2, 1:-1] >= center) * 2
    binary |= (img[:-2, 2:] >= center) * 4
    binary |= (img[1:-1, 2:] >= center) * 8
    binary |= (img[2:, 2:] >= center) * 16
    binary |= (img[2:, 1:-1] >= center) * 32
    binary |= (img[2:, :-2] >= center) * 64
    binary |= (img[1:-1, :-2] >= center) * 128
    return binary


def count_transitions(num):
    # Circular shift the number one position to the right
    shifted_num = (num >> 1) | ((num & 1) << 7)
    # XOR the original number with the shifted number
    xor_result = num ^ shifted_num
    # Count the number of 1s in the XOR result
    return bin(xor_result).count("1")


ulbp_58_9_codes = {}
for i in range(256):
    if count_transitions(i) <= 2:
        ulbp_58_9_codes[i] = bin(i).count("1")


def get_9ulbp_histogram(gray):
    lbp_code = get_lbp_code(gray)
    ulbp_histogram = np.zeros(9, dtype=np.uint32)
    for ulbp_58_code, ulbp_9_code in ulbp_58_9_codes.items():
        ulbp_histogram[ulbp_9_code] += np.sum(lbp_code == ulbp_58_code)
    return ulbp_histogram

def get_9ulbp_features(gray, cell_size):
    lbp_code = get_lbp_code(gray)
    height, width = gray.shape
    cell_height, cell_width = cell_size
    
    # Calculate the number of cells in each dimension
    num_cells_y = height // cell_height
    num_cells_x = width // cell_width
    
    # Initialize the LBP feature vector
    lbp_features = []
    
    # Iterate through the cells
    for y in range(num_cells_y):
        for x in range(num_cells_x):
            # Calculate the cell boundaries
            y_start, y_end = y * cell_height, (y + 1) * cell_height
            x_start, x_end = x * cell_width, (x + 1) * cell_width
            
            # Extract the cell's LBP code
            cell_lbp_code = lbp_code[y_start:y_end, x_start:x_end]
            
            # Calculate the 9ULBP histogram for the cell
            ulbp_histogram = np.zeros(9, dtype=np.uint32)
            for ulbp_58_code, ulbp_9_code in ulbp_58_9_codes.items():
                ulbp_histogram[ulbp_9_code] += np.sum(cell_lbp_code == ulbp_58_code)

            # Normalize the histogram (L2 norm)
            ulbp_histogram = ulbp_histogram / np.sqrt(np.sum(ulbp_histogram ** 2))

            # Append the cell's histogram to the LBP feature vector
            lbp_features.extend(ulbp_histogram)

    return np.array(lbp_features)