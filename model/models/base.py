from abc import ABC, abstractmethod
import joblib


class BaseModel(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def train(self, X, y) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self, X) -> int:
        """

        """
        ...

    @abstractmethod
    def data_transform(self, Z) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self

    def save(self, path) -> None:
        joblib.dump(self, path)

    def load(self, path) -> None:
        joblib.load(self, path)