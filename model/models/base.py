from abc import ABC, abstractmethod
import joblib

from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.info_logger import InfoLogger


class BaseModel(ABC):
    logger: AbstractLogger
    def __init__(self) -> None:
        self.model = None
        self.logger = InfoLogger("BaseModel")
        ...

    @abstractmethod
    def train(self, X, y) -> None:
        """
        Train the model using ML Models for multi-class and multi-label classification.
        :params: X, y is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self, X) -> list:
        """
        Make prediction using the trained ML Model for multi-class and multi-label classification.
        :params: X is essential, others are model specific
        :return: prediction
        """
        ...

    def save(self, path) -> None:
        joblib.dump(self.model, path)

    def load(self, path) -> None:
        self.model = joblib.load(path)

    def __str__(self):
        return str(self.model)