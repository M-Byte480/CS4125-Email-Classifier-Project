# Parent Classification Classes
from abc import ABC
from typing import override

from sklearn.metrics import accuracy_score

from model.classification_strategy_interface import IClassificationStrategy
from model.models.base import BaseModel


class Classifier(IClassificationStrategy, ABC):
    __slots__ = ('model',)

    def __init__(self):
        self.model: BaseModel = None

    @override
    def train(self, X, y):
        self.model.train(X, y)

    @override
    def classify(self, email) -> str:
        prediction = self.model.predict([email])
        return prediction[0]

    @override
    def evaluate(self, X, y) -> float:
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy}")
        return accuracy

    @override
    def save(self, file_path):
        self.model.save(file_path)

    @override
    def load(self, file_path):
        self.model.load(file_path)

    def __str__(self):
        return str(self.model)