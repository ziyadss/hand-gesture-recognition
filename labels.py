import json
import os

from constants import LABELS_FILENAME, MEN_DIRECTORY, WOMEN_DIRECTORY

directories = [MEN_DIRECTORY, WOMEN_DIRECTORY]

corrupted_images = {
    "data/men/3/3_men (141).JPG",
    "data/men/3/3_men (140).JPG",
    "data/men/4/4_men (5).JPG",
    "data/men/4/4_men (6).JPG",
    "data/men/2/2_men (108).JPG",
    "data/men/2/2_men (107).JPG",
}

mislabelled_images = {
    "data/men/3/3_men (83).JPG": 4,
    "data/men/3/3_men (84).JPG": 4,
}

images = []

for directory in directories:
    for label_dir in os.listdir(directory):
        label = int(label_dir)

        label_dir_path = os.path.join(directory, label_dir)

        for filename in os.listdir(label_dir_path):
            image_path = os.path.join(label_dir_path, filename)

            if image_path in corrupted_images:
                continue

            if not image_path.endswith(".jpg") and not image_path.endswith(".JPG"):
                continue

            if image_path in mislabelled_images:
                label = mislabelled_images[image_path]

            image = {"path": image_path, "label": label}

            images.append(image)

with open(LABELS_FILENAME, "w") as f:
    json.dump(images, f)

print(f"Saved {len(images)} images")
