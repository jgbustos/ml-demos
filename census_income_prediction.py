# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from os.path import normpath, join, dirname
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def full_path(filename):
    """Returns the full normalised path of a file when working dir is the one containing this module."""
    return normpath(join(dirname(__file__), filename))

model = None
columns = None

def ready():
    """Returns whether the ML trained model has been loaded from file correctly."""
    return model is not None

def init():
    """Loads the ML trained model (plus ancillary files) from file."""
    global model, columns
    if not ready():
        model = joblib.load(full_path("models/XGBClassifier.pkl"))
        columns = joblib.load(full_path("models/columns.pkl"))

def run(data):
    """Makes a prediction using the trained ML model."""
    test = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, index=[0])
    test = pd.get_dummies(test)
    test = test.reindex(columns=columns, fill_value=0)
    prediction = model.predict(test)
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    return prediction

def sample():
    """Returns a sample input vector as a dictionary."""
    return {
        "age":43,
        "workclass":"Private",
        "fnlwgt":100000,
        "education":"Bachelors",
        "education-num":13,
        "marital-status":"Married-civ-spouse",
        "occupation":"Sales",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":40,
        "native-country":"Spain"
    }

if __name__ == "__main__":
    init()
    print(sample())
    print(run(sample()))
