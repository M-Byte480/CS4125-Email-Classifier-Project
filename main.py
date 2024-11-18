#This is a main file: The controller. All methods will directly on directly be called here
import numpy as np

from Config import Config
from data.preprocessing.processor import DataProcessor
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    return DataProcessor.load_data()


def preprocess_data(data_frame):
    # De-duplicate input data
    data_frame =  DataProcessor.de_duplication(data_frame)
    # Translate
    data_frame = DataProcessor.translate_loaded_data(data_frame)
    # remove noise in input data
    data_frame = DataProcessor.remove_noise(data_frame)
    # translate data to english
    data_frame[Config.TICKET_SUMMARY] = translate_to_en(data_frame[Config.TICKET_SUMMARY].tolist())
    return data_frame

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

def classification_context(classification_strategy: str):
    classification_context = ClassificationContextFactory.create_context(classification_strategy)
    return classification_context

# Code will start executing from following line
if __name__ == '__main__':
    
    # pre-processing steps
    data_frame = load_data()
    data_frame = preprocess_data(data_frame)
    data_frame[Config.INTERACTION_CONTENT] = data_frame[Config.INTERACTION_CONTENT].values.astype('U')
    data_frame[Config.TICKET_SUMMARY] = data_frame[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(data_frame)
    logistic_model = classification_context('logistic_regression')

    # data modelling
    data = get_data_object(X, data_frame)
    # modelling
    perform_modelling(data, data_frame, 'name')

