# This is a main file: The controller. All methods will directly on directly be called here
import numpy as np
import random
import os
import sys
import pandas as pd

from preprocessing.processor import DataProcessor
from model.classification import *
from utilities.utility import convert_y_values, load_values, instantiate_all_models, exists_file, get_best_model, \
    train_models, extract_training_data, load_data

seed = 0
random.seed(seed)
np.random.seed(seed)

def pre_process_email(email_text):
    return email_text

def predict_email(list_of_classification_contexts, email_text):
    classification_type = {}
    for label_name, classification_context in list_of_classification_contexts.items():
        classification_type[label_name] = classification_context.classify_email(email_text)
    return classification_type


if __name__ == '__main__':

    args = sys.argv
    # Loading data frame and preprocessing if needed
    app_data_frame = load_values(DataProcessor.PATH_TO_APP_PREPROCESSED)
    pur_data_frame_ = load_values(DataProcessor.PATH_TO_PURCHASES_PREPROCESSED)

    combined_data_frame = pd.concat([app_data_frame, pur_data_frame_], axis=0, ignore_index=True)
    X, y = extract_training_data(app_data_frame)
    # X_pur, y_pur = extract_training_data(pur_data_frame_)


    # models = instantiate_all_models()
    
    # X, y = convert_y_values(X, y)

    # todo: for type 3 and 4 we need to remove null labels as they can't be used in training
    # Furthermore this should be done for every type in the case that a null label exists as we will never be able to train on unlabelled data with supervised learning
    # Training models and evaluating the best model for each label

    # Iterate over each label type (y1, y2, etc.) and their values
    for idx, (label_name, y_val) in enumerate(y.items()):
        print(f"Processing label type: {label_name}")

        # Generate file path for saving/loading model

        # -t to train
        if "-t" in args:
            print(f"Training models for {label_name}...")
            # Instantiate fresh models for this task
            models = instantiate_all_models()
            y_val = y_val.astype(str)

            # Train models on the current task
            train_models(models, X, y_val)

            # Get the best-performing model for this task
            best_model = get_best_model(models, X, y_val)

            os.makedirs("trained_models", exist_ok=True)

            # Save the trained model
            model_path = f"trained_models/{label_name}_model.{str(best_model)}"

            print(f"Saving model to {model_path}")
            best_model.save_model(model_path)

    best_models = list()

    list_of_models = os.listdir("trained_models")
    list_of_classification_contexts = {}
    for idx, (label_name, y_val) in enumerate(y.items()):
        model_path = [model for model in list_of_models if model.startswith(label_name)][0]

        print(f"Loading model from {model_path}")

        classification_context: ClassificationContext = ClassificationContextFactory.create_context(model_path.split(".")[1])
        classification_context._strategy.load(os.path.join("trained_models", model_path))
        list_of_classification_contexts[label_name] = classification_context

    if "-e" in args:
        email_id = int(args[args.index("-e") + 1])
        email_frame = load_data("sample_emails.csv")
        email = email_frame.iloc(email_id)

        processed_email = pre_process_email(email)

        classification_type = predict_email(list_of_classification_contexts, processed_email)

        print("Classifying email:", email)
        print("Results:")
        for label, classification in classification_type.items():
            print(f"{label}: {classification}")


