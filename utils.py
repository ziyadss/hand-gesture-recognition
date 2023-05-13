import json
from typing import Callable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import EXAMPLES_PER_LABEL, LABELS_FILENAME


def show_cv2_image_bgr(img):
    print_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(print_img)
    plt.show()


def show_cv2_image_gray(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def load_data(
    features_function: Callable[[str], np.ndarray]
) -> tuple[list[np.ndarray], list[int]]:
    data: list[np.ndarray] = []
    labels: list[int] = []

    with open(LABELS_FILENAME, "r") as f:
        json_data = json.load(f)

    if EXAMPLES_PER_LABEL is None:
        count = 1821
        i = 0
        for entry in json_data:
            image_path: str = entry["path"]
            label: int = entry["label"]

            features = features_function(image_path)

            data.append(features)
            labels.append(label)

            if i % 50 == 0:
                print(f"Loaded {i}/{count} images")
            i += 1
    else:
        count = 6 * EXAMPLES_PER_LABEL
        i = 0
        samples_per_class: dict[int, int] = {}

        for entry in json_data:
            image_path: str = entry["image_url"]
            label: int = entry["label"]

            if label not in samples_per_class:
                samples_per_class[label] = 0
            if samples_per_class[label] >= EXAMPLES_PER_LABEL:
                continue
            samples_per_class[label] += 1

            features = features_function(image_path)

            data.append(features)
            labels.append(label)

            if i % 50 == 0:
                print(f"Loaded {i}/{count} images")
            i += 1
    return data, labels


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
