# This file contains some variable names you need to use in overall project. 
#For example, this will contain the name of dataframe columns we will working on each file
class Config:
    TRAINED_MODELS_DIR = 'trained_models'

    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['Type 3', 'Type 4']
    CLASS_COL = 'Type 2'
    GROUPED = 'Type 1'

    # Where preprocessed data is saved
    PREPROCESSED_DATA_PATH = 'data/preprocessed_data/preprocessed.csv'