import json
import os
import pickle
import time

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC

from constants import *
from extractor import get_features_from_path
from utils import NumpyEncoder, load_data

t_start = time.perf_counter()

DATA_FILENAME = "data_train.pkl"
MODEL_FILENAME = "model_train.pkl"
SCORES_FILENAME = "scores_train.json"

extract_data = FORCE_EXTRACT or not os.path.exists(DATA_FILENAME)

if extract_data:
    data, labels = load_data(get_features_from_path)
    with open(DATA_FILENAME, "wb") as f:
        pickle.dump((data, labels), f)
else:
    with open(DATA_FILENAME, "rb") as f:
        data, labels = pickle.load(f)

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


with open(SCORES_FILENAME, "w") as f:
    json.dump(scores, f, cls=NumpyEncoder, indent=4)

clf.fit(X_train, y_train)

with open(MODEL_FILENAME, "wb") as f:
    pickle.dump(clf, f)

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.1f}%")

t_end = time.perf_counter()
seconds = t_end - t_start

print(f"Time: {seconds:.1f} seconds")
