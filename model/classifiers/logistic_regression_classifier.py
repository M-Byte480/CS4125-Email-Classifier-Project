from model.classification_context import Classifier
from model.models.logistic_regression import LogisticRegressionModel


class LogisticRegressionClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegressionModel()