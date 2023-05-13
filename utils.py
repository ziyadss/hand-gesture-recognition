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

    if EXAMPLES_PER_LABEL is None:
        with open(LABELS_FILENAME, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                image_path: str = entry["image_url"]
                label: int = int(entry["label"])

                features = features_function(image_path)

                data.append(features)
                labels.append(label)
    else:
        samples_per_class: dict[int, int] = {}
        with open(LABELS_FILENAME, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                image_path: str = entry["image_url"]
                label: int = int(entry["label"])

                if label not in samples_per_class:
                    samples_per_class[label] = 0
                if samples_per_class[label] >= EXAMPLES_PER_LABEL:
                    continue
                samples_per_class[label] += 1

                features = features_function(image_path)

                data.append(features)
                labels.append(label)
    return data, labels


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
