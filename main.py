# This is a main file: The controller. All methods will directly on directly be called here
import os
import sys
import pandas as pd

from observers.results_displayer import ResultsDisplayer
from observers.statistics_collector import StatisticsCollector
from preprocessing.processor import DataProcessor
from model.classification import *
from utilities.utility import load_values, instantiate_all_models, get_best_model, train_models, load_data
from Config import Config

def print_usage() -> None:
    print("""
Usage: python main.py
        -t <model-name>       : Trains the specified model for all labels. These models are not saved and are for one time use.
        -l                    : Lists all trainable models.
        -r                    : Trains the best model for each label and saves the models for future use unless another model is specified. This will overwrite any previously saved models.
        -u                    : Use saved models for classification. If insufficient saved models exist this will return an error.
        -c <path/to/file.csv> : Classifies emails in the file at the specified location (trained models are required for this to work).
Make sure to specify only one of -t, -r, or -u, otherwise you may overwrite previously loaded models!""")

def load_training_data():
    app_data_frame = load_values(DataProcessor.PATH_TO_APP_PREPROCESSED)
    pur_data_frame_ = load_values(DataProcessor.PATH_TO_PURCHASES_PREPROCESSED)
    combined_data_frame = pd.concat([app_data_frame, pur_data_frame_], axis=0, ignore_index=True)
    return processor.vectorize_data(combined_data_frame)

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
    # We need to use the same processor with the same vectoriser
    processor = DataProcessor()

    # Train a specific model
    if '-t' in args:
        # If model not specified exit
        if args[args.index('-t') + 1] is None:
            print("Model name not found!")
            print_usage()
            exit(1)

        model_name = str(args[args.index('-t') + 1])
        X, y = load_training_data()

        for label_name, y_val in y.items():
            print(f"Training model for {label_name}...")
            try:
                model_context = ClassificationContextFactory.create_context(model_name)
            except ValueError as e:
                print("Model name invalid!")
                exit(1)

            # Remove unlabelled rows
            X_trimmed, y_val = DataProcessor.remove_nan_rows(X, y_val)
            y_val = y_val.astype(str)

            # Train models on the current task
            model_context.train_model(X_trimmed, y_val)

            # Add model
            models.append(model_context)

    # Train best performing models for each label and save them
    if '-r' in args:
        # Loading data and preprocessing if needed
        X, y = load_training_data()

        for label_name, y_val in y.items():
            print(f"Training models for {label_name}...")
            # Instantiate fresh models for this task
            candidates = instantiate_all_models()

            # Remove unlabelled rows
            X_trimmed, y_val = DataProcessor.remove_nan_rows(X, y_val)
            y_val = y_val.astype(str)

            # Train models on the current task
            train_models(candidates, X_trimmed, y_val)

            # Get the best-performing model for this task
            best_model = get_best_model(candidates, X_trimmed, y_val)
            models.append(best_model)

            os.makedirs(Config.TRAINED_MODELS_DIR, exist_ok=True)

            # Save the trained model
            model_path = f"trained_models/{label_name}_model.{str(best_model)}"

            print(f"Saving model to {model_path}")
            best_model.save_model(model_path)

    # Load pretrained models
    if '-u' in args:
        model_paths = os.listdir(Config.TRAINED_MODELS_DIR)

        if len(model_paths) < 4:
            print(f"Not enough saved models, expected {4}, found {len(model_paths)}")

        for model_path in model_paths:
            model_type = model_path.split('.')[1]
            model_context = ClassificationContextFactory.create_context(model_type)
            model_context.load_model(os.path.join(Config.TRAINED_MODELS_DIR, model_path))
            models.append(model_context)

        # We have to load/preprocess data to fit our vectoriser
        load_training_data()

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

    exit(0)