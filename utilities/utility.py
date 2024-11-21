import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from model.classification import ClassificationContext, ClassificationContextFactory
from preprocessing.processor import DataProcessor


def convert_y_values(X, y):
    # Step 1: Convert to DataFrame for convenience
    df = pd.DataFrame(y, columns=["y1", "y2", "y3", "y4"])

    encoders = {}
    encoded_columns = []

    for col in df.columns:
        encoder = OneHotEncoder(sparse=False)
        one_hot = encoder.fit_transform(df[[col]])
        encoded_columns.append(one_hot)
        encoders[col] = encoder

    y_encoded = np.hstack(encoded_columns)

    print(f"One-hot encoded Y:\n{y_encoded}")

def load_values(file_path):
    if exists_file(file_path):
        data_frame = load_data(file_path)
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
    else:
        file_path = DataProcessor.PATH_TO_APP if "App" in file_path else DataProcessor.PATH_TO_PURCHASES
        data_frame = load_data(file_path)
        data_frame = DataProcessor.renaming_cols(data_frame)
        data_frame = preprocess_data(data_frame)
        # Save preprocessed data frame for reuse
        save_data(file_path, data_frame)

        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")

    return data_frame

def load_data(file_path):
    #load the input data
    return DataProcessor.load_data(file_path)

def save_data(file_path, data):
    return DataProcessor.save_data(file_path, data)

def exists_file(file_path) -> bool:
    """Checks if the specified file exists"""
    return os.path.isfile(file_path)

def preprocess_data(data_frame):
    # De-duplicate input data
    data_frame =  DataProcessor.de_duplication(data_frame)
    data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
    data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
    # Translate
    data_frame = DataProcessor.translate_data_frame(data_frame)
    # remove noise in input data
    data_frame = DataProcessor.remove_noise(data_frame)
    return data_frame

def instantiate_all_models() -> [ClassificationContext]:
    models = []
    models.append(ClassificationContextFactory.create_context("naive_bayes"))
    models.append(ClassificationContextFactory.create_context("decision_tree"))
    models.append(ClassificationContextFactory.create_context("random_forest"))
    models.append(ClassificationContextFactory.create_context("logistic_regression"))
    models.append(ClassificationContextFactory.create_context("svm"))
    models.append(ClassificationContextFactory.create_context("k_nearest_neighbors"))
    return models

def train_models(models: [ClassificationContext], X, y) -> None:
    for model in models:
        unique_classes = np.unique(y)
        if len(unique_classes) < 2 and model._strategy.__class__.__name__ in ["NaiveBayesClassifier", "LogisticRegressionClassifier", "SVMClassifier"]:
            print(f"Skipping training for {model}: only one class present ({unique_classes[0]})")
            continue
        model.train_model(X, y)

def get_best_model(models: [ClassificationContext], X, y) -> ClassificationContext:
    best_model_score = 0
    best_model = None
    for model in models:
        unique_classes = np.unique(y)
        if len(unique_classes) < 2 and model._strategy.__class__.__name__ in ["NaiveBayesClassifier",
                                                                              "LogisticRegressionClassifier",
                                                                              "SVMClassifier"]:
            print(f"Skipping Validation for {model}: only one class present ({unique_classes[0]})")
            continue
        model_score = model.evaluate_model(X, y)
        if model_score > best_model_score:
            best_model_score = model_score
            best_model = model

    return best_model

# Code will start executing from following line
def get_classifications(data_frame):
    pass