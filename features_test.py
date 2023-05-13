import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC

from utils import get_features_from_path

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.jsonl")

RANDOM_STATE = 42

N = 10

t_start = time.process_time_ns()

data = []
labels = []
samples_per_class = {}

with open(LABELS_FILENAME, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        image_path = entry["image_url"]
        label = entry["label"]

        if label not in samples_per_class:
            samples_per_class[label] = 0
        if samples_per_class[label] >= N:
            continue
        samples_per_class[label] += 1

        features = get_features_from_path(image_path)

        data.append(features)
        labels.append(label)

with open("data_feat_test.pkl", "wb") as f:
    pickle.dump(data, f)

# with open("data_feat_test.pkl", "rb") as f:
#     data = pickle.load(f)

with open("labels_test.pkl", "wb") as f:
    pickle.dump(labels, f)

# with open("labels_test.pkl", "rb") as f:
#     labels = pickle.load(f)

# turn data entries into numpy arrays - they're either floats or tuples of floats, flatten the tuples
dict_data = data
data = []
for entry in dict_data:
    data.append([])
    for key in entry:
        if isinstance(entry[key], tuple):
            data[-1].extend(entry[key])
        else:
            data[-1].append(entry[key])
data = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE
)

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


with open("scores_playground_test.json", "w") as f:
    json.dump(scores, f, cls=NumpyEncoder, indent=4)

clf.fit(X_train, y_train)

with open("model_playground_test.pkl", "wb") as f:
    pickle.dump(clf, f)

# with open("model_playground_test.pkl", "rb") as f:
#     clf = pickle.load(f)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.1f}%")

t_end = time.process_time_ns()
print(f"Time: {(t_end - t_start) / 1e9:.1f} seconds")
