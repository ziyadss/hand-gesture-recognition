import os

FORCE_EXTRACT = False

RANDOM_STATE = 42
EXAMPLES_PER_LABEL = 20  # None for entire dataset

DATA_DIRECTORY = "data"
MEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "men")
WOMEN_DIRECTORY = os.path.join(DATA_DIRECTORY, "women")
LABELS_FILENAME = os.path.join(DATA_DIRECTORY, "labels.jsonl")
