import os
import pickle
import sys
import time

from sklearn.svm import SVC

from extractor import get_features_from_path

MODEL_FILENAME = "model_train.pkl"
with open(MODEL_FILENAME, "rb") as f:
    clf: SVC = pickle.load(f)

if len(sys.argv) != 2:
    print("Usage: python test.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]
paths = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if file.endswith(".jpg") or file.endswith(".JPG")
]

t_start = time.perf_counter()

data = [get_features_from_path(path) for path in paths]

predictions = clf.predict(data)  # type: ignore

with open("predictions.csv", "w") as f:
    for path, prediction in zip(paths, predictions):
        f.write(f"{path},{prediction}\n")

t_end = time.perf_counter()
seconds = t_end - t_start

print(f"Time: {seconds:.1f} seconds for {len(data)} images")
