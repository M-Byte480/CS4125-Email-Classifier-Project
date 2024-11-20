# This is a main file: The controller. All methods will directly on directly be called here
import numpy as np
import random
import os

from preprocessing.processor import DataProcessor
from model.classification import *

seed = 0
random.seed(seed)
np.random.seed(seed)


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

def extract_training_data(data_frame):
    return DataProcessor.vectorize_data(data_frame)

def instantiate_all_models() -> [ClassificationContext]:
    models = []
    models.append(ClassificationContextFactory.create_context("naive_bayes"))
    # models.append(ClassificationContextFactory.create_context("svm"))
    # models.append(ClassificationContextFactory.create_context("decision_tree"))
    # models.append(ClassificationContextFactory.create_context("random_forest"))
    # models.append(ClassificationContextFactory.create_context("logistic_regression"))
    return models

def train_models(models: [ClassificationContext], X, y) -> None:
    for model in models:
        model.train_model(X, y)

def get_best_model(models: [ClassificationContext], X, y) -> ClassificationContext:
    best_model_score = 0
    best_model = None
    for model in models:
        model_score = model.evaluate_model(X, y)
        if (model_score > best_model_score):
            best_model_score = model_score
            best_model = model
    return best_model

# Code will start executing from following line
# todo: for each label for each model train and store best for each label then add check if models already exist
def get_classifications(data_frame):
    pass

if __name__ == '__main__':

    # Loading data frame and preprocessing if needed
    if exists_file(DataProcessor.PATH_TO_APP_PREPROCESSED):
        data_frame = load_data(DataProcessor.PATH_TO_APP_PREPROCESSED)
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
        X, y = extract_training_data(data_frame)
    else:
        data_frame = load_data(DataProcessor.PATH_TO_APP)
        data_frame = DataProcessor.renaming_cols(data_frame)
        data_frame = preprocess_data(data_frame)
        # Save preprocessed data frame for reuse
        save_data(DataProcessor.PATH_TO_APP_PREPROCESSED, data_frame)

        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ts")
        data_frame = DataProcessor.replace_nan_data_in_column(data_frame, "x_ic")
        X, y = extract_training_data(data_frame)

    models = instantiate_all_models()

    # todo: for type 3 and 4 we need to remove null labels as they can't be used in training
    # Furthermore this should be done for every type in the case that a null label exists as we will never be able to train on unlabelled data with supervised learning
    # Training models and evaluating the best model for each label
    for idx, y_val in enumerate(y):
        if (exists_file(f'trained_models/type_{idx+1}_model.model')):
            print(f'Loading model from trained_models/type_{idx + 1}_model.model')
            models[0].load_model(f'trained_models/type_{idx+1}_model.model')
        else:
            train_models(models, X, y_val)
            best_model = get_best_model(models, X, y_val)
            print(f'Saving model to trained_models/type_{idx+1}_model.model')
            best_model.save_model(f'trained_models/type_{idx+1}_model.model')

    # logistic_model = classification_context('logistic_regression')
    # svm_model = classification_context('svm')
    # random_forest_model = classification_context('random_forest')
    # knn_model = classification_context('knn')
    # decision_tree_model = classification_context('decision_tree')

    # Train mode

    # data modelling
    # data = get_data_object(X_IS, data_frame)
    # modelling
    # perform_modelling(data, data_frame, 'name')
    # """

