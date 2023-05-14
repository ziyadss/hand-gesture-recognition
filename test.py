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

files = sorted(
    (
        file
        for file in os.listdir(folder_path)
        if file.endswith(".jpg") or file.endswith(".JPG")
    ),
    key=lambda x: int(x.split(".")[0]),
)

paths = [os.path.join(folder_path, file) for file in files]

results = []
times = []

for path in paths:
    t_start = time.perf_counter()

    features = get_features_from_path(path)

    prediction = clf.predict([features])[0]  # type: ignore

    t_end = time.perf_counter()

    seconds = t_end - t_start
    results.append(prediction)
    times.append(seconds)


with open("results.txt", "w") as f:
    for r in results:
        f.write(f"{r}\n")

with open("times.txt", "w") as f:
    for t in times:
        f.write(f"{t:.3f}\n")

print("File order:")
print(files)
