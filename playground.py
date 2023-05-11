import json
import os
import pickle
import time

import numpy as np
from skimage import io, transform, color
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.jsonl")

RANDOM_STATE = 42

N = 100

t_start = time.process_time_ns()

data = []
labels = []

# with open(LABELS_FILENAME, "r") as f:
#     for line in f:
#         entry = json.loads(line.strip())
#         image_path = entry["image_url"]
#         label = entry["label"]

#         image = io.imread(image_path, as_gray=True) * 255
#         image = transform.rescale(
#             image, 1 / 32, anti_aliasing=True, preserve_range=True
#         )

#         # features = local_binary_pattern(image, 8, 1, method="uniform")
#         # n_bins = int(features.max() + 1)
#         # features, _ = np.histogram(features, bins=n_bins, range=(0, n_bins), density=True)
#         # features = np.zeros((256, 256))

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

# with open("data_hog.pkl", "wb") as f:
#     pickle.dump(data, f)

with open("data_hog.pkl", "rb") as f:
    data = pickle.load(f)

# with open("labels.pkl", "wb") as f:
#     pickle.dump(labels, f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC

# base_clf = SVC(random_state=RANDOM_STATE, kernel="rbf")
# clf = AdaBoostClassifier(estimator=base_clf,random_state=RANDOM_STATE, n_estimators=100, algorithm="SAMME")

clf = SVC(random_state=RANDOM_STATE, kernel="rbf")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

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
