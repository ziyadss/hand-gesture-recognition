# hand-gesture recognition

## Prerequisites

This project was developed and tested on Python 3.10.6. It is recommended to use a virtual environment to run this project.
To install the required packages, run the following command:

```
pip3 install -r requirements.txt
```

## To train the model, run the following command:

```
python3 train.py
```

## To run the model, run the following command:

```
python3 test.py <folder_path>
```

where <folder_path> is the path to the folder containing the (JPG format) images to be tested.
This will output `predictions.csv`, containing the label for each image.
