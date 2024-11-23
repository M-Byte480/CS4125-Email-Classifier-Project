# Parent Classification Classes
from abc import ABC

from sklearn.metrics import accuracy_score

from model.classification_strategy_interface import IClassificationStrategy
from model.models.base import BaseModel
from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class Classifier(IClassificationStrategy, ABC):
    __slots__ = ('model',)

    def __init__(self):
        self.model: BaseModel = None
        self.info_logger = InfoLogger()
        self.info_logger = PrefixLogger(self.info_logger, "Model")

    def train(self, X, y):
        self.model.train(X, y)

    def classify(self, email) -> str:
        prediction = self.model.predict([email])
        return prediction[0]

    def evaluate(self, X, y) -> float:
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred) * 100
        self.info_logger.log(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model.load(file_path)

    def __str__(self):
        return str(self.model)
