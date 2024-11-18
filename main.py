#This is a main file: The controller. All methods will directly on directly be called here
import numpy as np

from Config import Config
from preprocessing.processor import DataProcessor
from embedding import get_tfidf_embd
from modelling.modelling import *
from modelling.data_model import *
import random
import os
import pandas as pd
seed =0
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
if __name__ == '__main__':
    
    # pre-processing steps
    if exists_file(DataProcessor.PATH_TO_APP_PREPROCESSED):
        load_data(DataProcessor.PATH_TO_APP_PREPROCESSED)
    else:
        data_frame = load_data(DataProcessor.PATH_TO_APP)
        data_frame = preprocess_data(data_frame)
        save_data(data_frame, DataProcessor.PATH_TO_APP_PREPROCESSED)

    data_frame[Config.INTERACTION_CONTENT] = data_frame[Config.INTERACTION_CONTENT].values.astype('U')
    data_frame[Config.TICKET_SUMMARY] = data_frame[Config.TICKET_SUMMARY].values.astype('U')


    # data transformation
    X_IS, group_df = get_embeddings(data_frame, Config.INTERACTION_CONTENT)

    logistic_model = classification_context('logistic_regression')

    # data modelling
    data = get_data_object(X_IS, data_frame)
    # modelling
    perform_modelling(data, data_frame, 'name')
    # """

