import json
import os
import pickle
import time

import numpy as np
from skimage import color, io, transform
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

import utils

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.jsonl")

RANDOM_STATE = 42

N = 20

t_start = time.process_time_ns()

data = []
labels = []

# with open(LABELS_FILENAME, "r") as f:
#     for line in f:
#         entry = json.loads(line.strip())
#         image_path = entry["image_url"]
#         label = entry["label"]

#         hand, region, hull = utils.segment_hand(image_path)
#         bbox = region.bbox
#         image = hand[bbox[0] : bbox[2], bbox[1] : bbox[3]]

#         image = transform.resize(image, (128, 128), preserve_range=True)

#         features = hog(
#             image,
#             orientations=8,
#             pixels_per_cell=(16, 16),
#             cells_per_block=(1, 1),
#             block_norm="L2-Hys",
#             feature_vector=True,
#         )

#         data.append(features)
#         labels.append(label)

# with open("data_hog_segmented.pkl", "wb") as f:
#     pickle.dump(data, f)

with open("data_hog_segmented.pkl", "rb") as f:
    data = pickle.load(f)

# with open("labels.pkl", "wb") as f:
#     pickle.dump(labels, f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE
)

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC

# base_clf = SVC(random_state=RANDOM_STATE, kernel="rbf")
# clf = AdaBoostClassifier(estimator=base_clf,random_state=RANDOM_STATE, n_estimators=100, algorithm="SAMME")

clf = RandomForestClassifier(random_state=RANDOM_STATE)  # , kernel="rbf")
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_validate(
    clf,
    X_train,
    y_train,
    cv=cv,
    scoring=("accuracy", "balanced_accuracy"),
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open("scores_playground.json", "w") as f:
    json.dump(scores, f, cls=NumpyEncoder, indent=4)

clf.fit(X_train, y_train)

with open("model_playground.pkl", "wb") as f:
    pickle.dump(clf, f)

# with open("model_playground.pkl", "rb") as f:
#     clf = pickle.load(f)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.1f}%")

t_end = time.process_time_ns()
print(f"Time: {(t_end - t_start) / 1e9:.1f} seconds")
