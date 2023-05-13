import json
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC

from features import get_features_from_path

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.jsonl")

RANDOM_STATE = 42
EXAMPLES_PER_LABEL = 20

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
        if samples_per_class[label] >= EXAMPLES_PER_LABEL:
            continue
        samples_per_class[label] += 1

        features = get_features_from_path(image_path)

        data.append(features)
        labels.append(label)

with open("train_data.pkl", "wb") as f:
    pickle.dump((data, labels), f)

# To load already extracted data, comment the extraction above and uncomment this.
# with open("train_data.pkl", "rb") as f:
#     data, labels = pickle.load(f)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=RANDOM_STATE
)

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


with open("scores_train.json", "w") as f:
    json.dump(scores, f, cls=NumpyEncoder, indent=4)

clf.fit(X_train, y_train)

with open("model_train.pkl", "wb") as f:
    pickle.dump(clf, f)

# To load already trained model, comment the training above and uncomment this.
# with open("model_train.pkl", "rb") as f:
#     clf = pickle.load(f)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.1f}%")

t_end = time.process_time_ns()
print(f"Time: {(t_end - t_start) / 1e9 / 60:.1f} minutes")
