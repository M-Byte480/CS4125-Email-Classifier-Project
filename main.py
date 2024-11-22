# This is a main file: The controller. All methods will directly on directly be called here
import os
import sys
import pandas as pd

from observers.results_displayer import ResultsDisplayer
from observers.statistics_collector import StatisticsCollector
from preprocessing.processor import DataProcessor, VectoriserManager
from model.classification_context import *
from utilities.logger.error_logger import ErrorLogger
from utilities.utility import Utils
from utilities.configuration.config import Config

class Main:
    def __init__(self):
        self.utils = Utils()
        pass


    def print_usage(logger) -> None:
        logger.log("""Usage: python main.py
            -t <model-name>       : Trains the specified model for all labels. These models are not saved and are for one time use.
            -l                    : Lists all trainable models.
            -r                    : Trains the best model for each label and saves the models for future use unless another model is specified. This will overwrite any previously saved models.
            -u                    : Use saved models for classification. If insufficient saved models exist this will return an error.
            -c <path/to/file.csv> : Classifies emails in the file at the specified location (trained models are required for this to work).
    Make sure to specify only one of -t, -r, or -u, otherwise you may overwrite previously loaded models!""")

    def load_training_data(self):
        app_data_frame = self.utils.load_values(DataProcessor.PATH_TO_APP_PREPROCESSED)
        pur_data_frame_ = self.utils.load_values(DataProcessor.PATH_TO_PURCHASES_PREPROCESSED)
        combined_data_frame = pd.concat([app_data_frame, pur_data_frame_], axis=0, ignore_index=True)

        return combined_data_frame

    @staticmethod
    def main():
        main = Main()
        args = sys.argv
        error_logger = ErrorLogger("Main")
        logger = InfoLogger("Main")

        # Args are necessary
        if len(args) <= 1:
            main.print_usage(error_logger)
            exit(0)

        # List all trainable models
        if '-l' in args:
            logger.log("""Trainable models:
            - naive_bayes
            - svm
            - decision_tree
            - random_forest
            - logistic_regression
            - k_nearest_neighbors
        """)

        # This array is used to store trained models for classification
        models = []
        # Load training data
        df = main.load_training_data()
        vectoriser = VectoriserManager()
        vectoriser.fit_vectoriser(df["x_ic"])
        X, y = vectoriser.vectorize_data(df)

        # Train a specific model
        if '-t' in args:
            # If model not specified exit
            if args[args.index('-t') + 1] is None:
                error_logger.log("Model name not found!")
                main.print_usage(error_logger)
                exit(1)

            model_name = str(args[args.index('-t') + 1])

            for label_name, y_val in y.items():
                logger.log(f"Training model for {label_name}...")
                try:
                    model_context = ClassificationContextFactory.create_context(model_name)
                except ValueError as e:
                    error_logger.log("Model name invalid!")
                    exit(1)

                # Remove unlabelled rows
                X_trimmed, y_trim_val = DataProcessor.remove_nan_rows(X, y_val)
                y_trim_val = y_trim_val.astype(str)

                # Train models on the current task
                model_context.train_model(X_trimmed, y_trim_val)

                # Add model
                models.append(model_context)

        # Train best performing models for each label and save them
        if '-r' in args:
            for label_name, y_val in y.items():
                logger.log(f"Training models for {label_name}...")
                # Instantiate fresh models for this task
                candidates = Utils.instantiate_all_models()

                # Remove unlabelled rows
                X_trimmed, y_trim_val = DataProcessor.remove_nan_rows(X, y_val)
                y_trim_val = y_trim_val.astype(str)

                # Train models on the current task
                Utils.train_models(candidates, X_trimmed, y_trim_val)

                # Get the best-performing model for this task
                best_model = Utils.get_best_model(candidates, X_trimmed, y_trim_val)
                models.append(best_model)

                os.makedirs(Config.TRAINED_MODELS_DIR, exist_ok=True)

                # Save the trained model
                model_path = f"trained_models/{label_name}_model.{str(best_model)}"

                logger.log(f"Saving model to {model_path}")
                best_model.save_model(model_path)

        # Load pretrained models
        if '-u' in args:
            model_paths = os.listdir(Config.TRAINED_MODELS_DIR)

            if len(model_paths) < 4:
                error_logger.log(f"Not enough saved models, expected {4}, found {len(model_paths)}")
                exit(1)

            for model_path in model_paths:
                model_type = model_path.split('.')[1]
                model_context = ClassificationContextFactory.create_context(model_type)
                model_context.load_model(os.path.join(Config.TRAINED_MODELS_DIR, model_path))
                models.append(model_context)

        # Classify emails in the specified CSV
        if "-c" in args:
            # If file path not specified exit
            if args[args.index('-c') + 1] is None:
                main.print_usage(error_logger)
                exit(1)

            file_path = str(args[args.index("-c") + 1])

            try:
                email_df = DataProcessor.load_data(file_path)
                email_df = DataProcessor.renaming_cols(email_df)
                email_df = DataProcessor.translate_data_frame(email_df)
                X = vectoriser.vectorize_unclassified_data(email_df)
            except Exception as e:
                error_logger.log(e)
                error_logger.log("""
        Error in processing of data
        Make sure in your CSV you have the at least the following columns:
            - Ticket Summary
            - Interaction Content""")
                exit(1)

            logger.log(f"Classifying emails in {file_path}")
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


if __name__ == '__main__':
    Main.main()