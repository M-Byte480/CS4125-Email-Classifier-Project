# This is a main file: The controller. All methods will directly or indirectly be called here.
import os
import sys
import traceback

from model.factory.classification_factory import ClassificationContextFactory
from observers.results_displayer import ResultsDisplayer
from observers.statistics_collector import StatisticsCollector
from preprocessing.processor import DataProcessor, VectoriserManager
from utilities.logger.concrete_logger.error_logger import ErrorLogger
from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger
from utilities.utility import Utils
from utilities.configuration.config import Config
from utilities.file_manager import FileManager

class Main:
    def __init__(self):
        self.utils = Utils()
        pass

    def print_usage(self, logger) -> None:
        logger.log("""Usage: python main.py
            -t <model-name>       : Trains the specified model for all labels. These models are not saved and are for one time use.
            -l                    : Lists all trainable models.
            -r                    : Trains the best model for each label and saves the models for future use unless another model is specified. This will overwrite any previously saved models.
            -u                    : Use saved models for classification. If insufficient saved models exist this will return an error.
            -c <path/to/file.csv> : Classifies emails in the file at the specified location (trained models are required for this to work).
    Make sure to specify only one of -t, -r, or -u, otherwise you may overwrite previously loaded models!""")

    @staticmethod
    def main(args):
        main = Main()
        error_logger = ErrorLogger()
        error_logger = PrefixLogger(error_logger, "MAIN")
        logger = InfoLogger()
        logger = PrefixLogger(logger, "MAIN")
        file_manager = FileManager()

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

        # Load existing preprocessing data or preprocess training data
        if file_manager.exists_file(Config.PREPROCESSED_DATA_PATH):
            try:
                df = file_manager.load_csv(Config.PREPROCESSED_DATA_PATH)
            except FileNotFoundError as e:
                error_logger.log(str(e))
                error_logger.log(traceback.format_exc())
                exit(1)
            df = DataProcessor.replace_nan_data_in_column(df, "x_ts")
            df = DataProcessor.replace_nan_data_in_column(df, "x_ic")
        else:
            try:
                df = file_manager.load_all_csvs_in_directory("data/training_data")
            except FileNotFoundError as e:
                error_logger.log(str(e))
                error_logger.log(traceback.format_exc())
                exit(1)
            # Preprocess the training data
            df = DataProcessor.renaming_cols(df)
            df = DataProcessor.de_duplication(df)
            df = DataProcessor.replace_nan_data_in_column(df, "x_ts")
            df = DataProcessor.replace_nan_data_in_column(df, "x_ic")
            df = DataProcessor.translate_data_frame(df)
            df = DataProcessor.remove_noise(df)
            # Save the preprocessed data for re-use
            file_manager.save_csv(df, Config.PREPROCESSED_DATA_PATH)

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
            training_logger = PrefixLogger(logger, "TRAINING")
            for label_name, y_val in y.items():
                training_logger.log(f"Training model for {label_name}...")
                try:
                    model_context = ClassificationContextFactory.create_context(model_name)
                except ValueError as e:
                    error_logger.log(str(e))
                    error_logger.log("Model name invalid! Use -l to get a list of all trainable model names.")
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
                email_df = file_manager.load_csv(file_path)
                email_df = DataProcessor.renaming_cols(email_df)
                email_df = DataProcessor.translate_data_frame(email_df)
                X = vectoriser.vectorize_unclassified_data(email_df)
            except Exception as e:
                error_logger.log(str(e))
                error_logger.log(traceback.format_exc())
                error_logger.log("""Error in processing of data
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
    args = sys.argv
    Main.main(args)
