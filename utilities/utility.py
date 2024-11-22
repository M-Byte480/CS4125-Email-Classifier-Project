from model.classification_context import ClassificationContext
from model.factory.classification_factory import ClassificationContextFactory

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def instantiate_all_models() -> [ClassificationContext]:
        models = list()
        models.append(ClassificationContextFactory.create_context("naive_bayes"))
        models.append(ClassificationContextFactory.create_context("decision_tree"))
        models.append(ClassificationContextFactory.create_context("random_forest"))
        models.append(ClassificationContextFactory.create_context("logistic_regression"))
        models.append(ClassificationContextFactory.create_context("svm"))
        models.append(ClassificationContextFactory.create_context("k_nearest_neighbors"))
        return models

    @staticmethod
    def train_models(models: [ClassificationContext], X, y) -> None:
        for model in models:
            model.train_model(X, y)

    @staticmethod
    def get_best_model(models: [ClassificationContext], X, y) -> ClassificationContext:
        best_model_score = 0
        best_model = None
        for model in models:
            model_score = model.evaluate_model(X, y)
            if model_score > best_model_score:
                best_model_score = model_score
                best_model = model

        return best_model
