import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

import pandas as pd

# Sample mock dataset for testing
@pytest.fixture
def sample_data():
    data = {
        "age": [25, 38, 28, 44],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "Private"],
        "fnlwgt": [226802, 89814, 336951, 160323],
        "education": ["11th", "HS-grad", "Assoc-acdm", "Some-college"],
        "education-num": [7, 9, 12, 10],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Farming-fishing", "Protective-serv", "Machine-op-inspct"],
        "relationship": ["Own-child", "Husband", "Not-in-family", "Husband"],
        "race": ["Black", "White", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "capital-gain": [0, 0, 0, 7688],
        "capital-loss": [0, 0, 0, 0],
        "hours-per-week": [40, 50, 40, 40],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": ["<=50K", ">50K", ">50K", "<=50K"]
    }
    df = pd.DataFrame(data)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    return X, y, encoder, lb


def test_model_type(sample_data):
    """
    Test if train_model returns a RandomForestClassifier instance.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not RandomForestClassifier."


def test_inference_output_shape(sample_data):
    """
    Test if inference returns an array of the correct shape.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Predictions are not a numpy array."
    assert preds.shape == y.shape, "Prediction shape does not match label shape."


def test_metrics_output_values():
    """
    Test compute_model_metrics returns expected values on known inputs.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert np.isclose(precision, 1.0), f"Unexpected precision: {precision}"
    assert np.isclose(recall, 0.6667, atol=0.01), f"Unexpected recall: {recall}"
    assert np.isclose(f1, 0.8, atol=0.01), f"Unexpected F1: {f1}"