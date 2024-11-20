from abc import ABC, abstractmethod
from typing import override

from sklearn.metrics import accuracy_score

from model.models.base import BaseModel
from model.models.SVM import SVMModel
from model.models.decisiontree import DecisionTreeModel
from model.models.logistic_regression import LogisticRegressionModel
from model.models.naive_bayes import NaiveBayesModel
from model.models.randomforest import RandomForest
from observers.email_classification_observer import EmailClassificationObserver
from structs.objects import Email

# Strategy Pattern - Interface
class IClassificationStrategy(ABC):
    @abstractmethod
    def train(self, X, y):
        pass
    @abstractmethod
    def classify(self, email: Email):
        pass

# Parent Classification Classes
class Classifier(IClassificationStrategy, ABC):
    __slots__ = ('model',)

    def __init__(self):
        self.model: BaseModel = None

    @override
    def train(self, X, y):
        self.model.train(X, y)

    @override
    def classify(self, email) -> str:
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

    @override
    def evaluate(self, X, y) -> float:
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    @override
    def save(self, file_path):
        self.model.save(file_path)

    @override
    def load(self, file_path):
        self.model.load(file_path)

# Strategy Pattern - Concrete Classification Classes
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = NaiveBayesModel()

class SVMClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = SVMModel()

class DecisionTreeClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeModel()

class RandomForestClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = RandomForest()

class LogisticRegressionClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegressionModel()

# Strategy Pattern - Context
class ClassificationContext:
    strategy: Classifier
    _observers: [EmailClassificationObserver]  # Array of subscribed observers

    def __init__(self, strategy: Classifier) -> None:
        self._strategy = strategy
        self._observers = []

    def set_strategy(self, strategy: Classifier) -> None:
        """Allows switching the strategy dynamically."""
        self._strategy = strategy

    def train_model(self, X, y):
        """Trains a model using the classification strategy"""
        self._strategy.train(X, y)

    def evaluate_model(self, X, y) -> float:
        """Evaluates a model using the classification strategy and returns its accuracy"""
        return self._strategy.evaluate(X, y)

    def classify_email(self, email: Email) -> str:
        """Classifies an email using the current strategy."""
        return self._strategy.classify(email)

    def save_model(self, file_path):
        """Saves the model in the Classifier"""
        self._strategy.save(file_path)

    # todo: model that is loaded needs to be the same type of model as the strategy (Classifier)
    def load_model(self, file_path):
        """Loads a model into the Classifier"""
        self._strategy.load(file_path)

    def add_observer(self, observer: EmailClassificationObserver) -> None:
        """Subscribe an observer to this subject."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: EmailClassificationObserver) -> None:
        """Unsubscribe an observer from this subject."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(self, classification: str) -> None:
        """Notify observers of a classification."""
        for observer in self._observers:
            observer.update(classification)

# Factory Pattern - Context Factory
class ClassificationContextFactory:
    @staticmethod
    def create_context(strategy: str) -> ClassificationContext:
        constructor_selector = {
            "naive_bayes": NaiveBayesClassifier,
            "svm": SVMClassifier,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegressionClassifier
        }

        constructor = constructor_selector.get(strategy)

        if constructor:
            return ClassificationContext(constructor())
        else:
            raise ValueError("Invalid strategy")
