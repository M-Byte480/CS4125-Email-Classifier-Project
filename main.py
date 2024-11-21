# This is a main file: The controller. All methods will directly on directly be called here
import numpy as np
import random
import os
import sys
import pandas as pd

from observers.results_displayer import ResultsDisplayer
from observers.statistics_collector import StatisticsCollector
from preprocessing.processor import DataProcessor
from model.classification import *
from utilities.utility import load_values, instantiate_all_models, get_best_model, train_models, load_data

seed = 0
random.seed(seed)
np.random.seed(seed)

def print_usage() -> None:
    print("""
Usage: python main.py
        -t <model-name>       : Trains the specified model for all labels. These models are not saved and are for one time use.
        -l                    : Lists all trainable models.
        -r                    : Trains the best model for each label and saves the models for future use unless another model is specified. This will overwrite any previously saved models.
        -c <path/to/file.csv> : Classifies emails in the file at the specified location (trained models are required for this to work).""")

if __name__ == '__main__':

    args = sys.argv

    # Args are necessary
    if len(args) <= 1:
        print_usage()
        exit(0)

    # List all trainable models
    if '-l' in args:
        print("""
Trainable models:
    - naive_bayes
    - svm
    - decision_tree
    - random_forest
    - logistic_regression
    - k_nearest_neighbors
""")

    # This array is used to store trained models for classification
    models = []
    processor = DataProcessor()

    # Train a specific model
    if '-t' in args:
        # If model not specified exit
        if args[args.index('-t') + 1] is None:
            print_usage()
            exit(1)
        model_name = str(args[args.index('-t') + 1])

        try:
            model_context = ClassificationContextFactory.create_context(model_name)
        except ValueError as e:
            print_usage()
            print(e)
            exit(1)

        # Loading data frame and preprocessing if needed
        app_data_frame = load_values(DataProcessor.PATH_TO_APP_PREPROCESSED)
        pur_data_frame_ = load_values(DataProcessor.PATH_TO_PURCHASES_PREPROCESSED)

        combined_data_frame = pd.concat([app_data_frame, pur_data_frame_], axis=0, ignore_index=True)
        X, y = processor.vectorize_data(combined_data_frame)

        for idx, (label_name, y_val) in enumerate(y.items()):
            print(f"Training model for {label_name}...")
            # Add model
            models.append(model_context)

            y_val = y_val.astype(str)

            # Train models on the current task
            train_models([model_context], X, y_val)

    # Train best performing models for each label and save them
    if '-r' in args:
        app_data_frame = load_values(DataProcessor.PATH_TO_APP_PREPROCESSED)
        pur_data_frame_ = load_values(DataProcessor.PATH_TO_PURCHASES_PREPROCESSED)

        combined_data_frame = pd.concat([app_data_frame, pur_data_frame_], axis=0, ignore_index=True)
        X, y = processor.vectorize_data(combined_data_frame)

        for idx, (label_name, y_val) in enumerate(y.items()):
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

    # Classify emails in the specified CSV
    if "-c" in args:
        # If file path not specified exit
        if args[args.index('-c') + 1] is None:
            print_usage()
            exit(1)

        file_path = str(args[args.index("-c") + 1])

        try:
            email_df = load_data(file_path)
            email_df = DataProcessor.renaming_cols(email_df)
            email_df = DataProcessor.translate_data_frame(email_df)
            X = processor.vectorize_unclassified_data(email_df)
        except Exception as e:
            print(e)
            print("""
Error in processing of data
Make sure in your CSV you have the at least the following columns:
    - Ticket Summary
    - Interaction Content""")
            exit(1)

        print(f"Classifying emails in {file_path}")
        # Subscribe observers for tracking classification information
        rd = ResultsDisplayer()
        sc = StatisticsCollector()
        for model in models:
            model.add_observer(rd)
            model.add_observer(sc)

        for idx, email in enumerate(X):
            for model in models:
                model.classify_email(email, email_df["x_ts"][idx], email_df["x_ic"][idx])

        sc.display_stats()
