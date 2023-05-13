import os

FORCE_EXTRACT = True

RANDOM_STATE = 465
EXAMPLES_PER_LABEL = None  # None for entire dataset

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.json")
