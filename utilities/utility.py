import os

import numpy as np

from model.classification_context import ClassificationContext
from model.factory.classification_factory import ClassificationContextFactory
from preprocessing.processor import DataProcessor
from utilities.logger.error_logger import ErrorLogger
from utilities.logger.warning_logger import WarningLogger


class Utils:
    def __init__(self):
        pass

    def load_values(self, file_path):
        if self.exists_file(file_path):
            data_frame = DataProcessor.load_data(file_path)
            data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
            data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
        else:
            file_path = DataProcessor.PATH_TO_APP if "App" in file_path else DataProcessor.PATH_TO_PURCHASES
            data_frame = DataProcessor.load_data(file_path)
            data_frame = DataProcessor.renaming_cols(data_frame)
            data_frame = self.preprocess_data(data_frame)
            # Save preprocessed data frame for reuse
            DataProcessor.save_data(file_path, data_frame)

            data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
            data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")

        return data_frame

    def exists_file(self, file_path) -> bool:
        """Checks if the specified file exists"""
        return os.path.isfile(file_path)

    def preprocess_data(self, data_frame):
        # De-duplicate input data
        data_frame =  DataProcessor.de_duplication(data_frame)
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
        # Translate
        data_frame = DataProcessor.translate_data_frame(data_frame)
        # remove noise in input data
        data_frame = DataProcessor.remove_noise(data_frame)
        return data_frame

    @staticmethod
    def instantiate_all_models() -> [ClassificationContext]:
        models = []
        models.append(ClassificationContextFactory.create_context("naive_bayes"))
        models.append(ClassificationContextFactory.create_context("decision_tree"))
        models.append(ClassificationContextFactory.create_context("random_forest"))
        models.append(ClassificationContextFactory.create_context("logistic_regression"))
        models.append(ClassificationContextFactory.create_context("svm"))
        models.append(ClassificationContextFactory.create_context("k_nearest_neighbors"))
        return models

    @staticmethod
    def train_models(models: [ClassificationContext], X, y) -> None:
        for model in models:
            model.train_model(X, y)

    @staticmethod
    def get_best_model(models: [ClassificationContext], X, y) -> ClassificationContext:
        best_model_score = 0
        best_model = None
        for model in models:
            model_score = model.evaluate_model(X, y)
            if model_score > best_model_score:
                best_model_score = model_score
                best_model = model

        return best_model
