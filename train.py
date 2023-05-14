import json
import os
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from constants import FORCE_EXTRACT, RANDOM_STATE
from extractor import get_features_from_path
from utils import NumpyEncoder, load_data

t_start = time.perf_counter()

DATA_FILENAME = "data_train.pkl"
MODEL_FILENAME = "model_train.pkl"
SCORES_FILENAME = "scores_train.json"

data_exists = os.path.exists(DATA_FILENAME)

if FORCE_EXTRACT or not data_exists:
    if data_exists:
        os.rename(DATA_FILENAME, DATA_FILENAME + ".bak")

    data, labels = load_data(get_features_from_path)
    with open(DATA_FILENAME, "wb") as f:
        pickle.dump((data, labels), f)
else:
    with open(DATA_FILENAME, "rb") as f:
        data, labels = pickle.load(f)

clf = SVC(random_state=RANDOM_STATE, kernel="poly", C=9, degree=8)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_validate(
    clf,
    data,  # type: ignore
    labels,
    cv=cv,
    scoring=("accuracy", "balanced_accuracy"),
)


with open(SCORES_FILENAME, "w") as f:
    json.dump(scores, f, cls=NumpyEncoder, indent=4)

clf.fit(data, labels)  # type: ignore

with open(MODEL_FILENAME, "wb") as f:
    pickle.dump(clf, f)

t_end = time.perf_counter()
seconds = t_end - t_start

print(f"Time: {seconds:.1f} seconds")
