import pickle
import time

from sklearn.svm import SVC

from extractor import get_features_from_path

t_start = time.perf_counter()

MODEL_FILENAME = "model_train.pkl"
with open(MODEL_FILENAME, "rb") as f:
    clf: SVC = pickle.load(f)


paths = []

data = [get_features_from_path(path) for path in paths]

predictions = clf.predict(data)  # type: ignore

with open("predictions.csv", "w") as f:
    for path, prediction in zip(paths, predictions):
        f.write(f"{path},{prediction}\n")

t_end = time.perf_counter()
seconds = t_end - t_start

print(f"Time: {seconds:.1f} seconds")
