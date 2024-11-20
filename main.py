#This is a main file: The controller. All methods will directly on directly be called here
import numpy as np
import random
import os
import pandas as pd

from Config import Config
from preprocessing.processor import DataProcessor
from embedding import get_tfidf_embd
from modelling.modelling import *
from modelling.data_model import *

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
    data_frame = DataProcessor.replace_nan_interaction_summary(data_frame)
    # Translate
    data_frame = DataProcessor.translate_data_frame(data_frame)
    # remove noise in input data
    data_frame = DataProcessor.remove_noise(data_frame)
    return data_frame

def get_embeddings(df:pd.DataFrame, text_column: str = Config.INTERACTION_CONTENT):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return extract_classifications(X, df)

def perform_modelling(data, df: pd.DataFrame, name):
    model_predict(data, df, name)

def classification_context(classification_strategy: str):
    classification_context = ClassificationContextFactory.create_context(classification_strategy)
    return classification_context

# Code will start executing from following line
# todo: for each label for each model train and store best for each label then add check if models already exist
def get_classifications(data_frame):
    pass

def extract_training_data(data_frame):
    return DataProcessor.vectorize_data(data_frame)

if __name__ == '__main__':
    
    # pre-processing steps
    if exists_file(DataProcessor.PATH_TO_APP_PREPROCESSED):
        data_frame = load_data(DataProcessor.PATH_TO_APP_PREPROCESSED)
        X, y = extract_training_data(data_frame)
    else:
        data_frame = load_data(DataProcessor.PATH_TO_APP)
        data_frame = DataProcessor.renaming_cols(data_frame)
        data_frame = preprocess_data(data_frame)
        X, y = extract_training_data(data_frame)
        save_data(data_frame, DataProcessor.PATH_TO_APP_PREPROCESSED)

    # data transformation

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

