from abc import ABC, abstractmethod
import joblib

class BaseModel(ABC):
    def __init__(self) -> None:
        self.model = None
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
        :return: prediction score
        """
        ...

    def save(self, path) -> None:
        joblib.dump(self.model, path)

    def load(self, path) -> None:
        self.model = joblib.load(path)

    def __str__(self):
        return str(self.model)