# Strategy Pattern - Interface
from abc import ABC, abstractmethod

class IClassificationStrategy(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    @abstractmethod
    def classify(self, email: Email):
        pass
    @abstractmethod
    def evaluate(self, X, y):
        pass
    @abstractmethod
    def save(self, file_path):
        pass
    @abstractmethod
    def load(self, file_path):
        pass