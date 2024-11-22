from model.classifier import Classifier

from observers.email_classification_observer import EmailClassificationObserver
from utilities.logger.indentation_decorator import IndentationDecorator
from utilities.logger.info_logger import InfoLogger

# Strategy Pattern - Context
class ClassificationContext:
    strategy: Classifier
    logger: InfoLogger
    _observers: [EmailClassificationObserver]  # Array of subscribed observers

    def __init__(self, strategy: Classifier) -> None:
        self._strategy = strategy
        self._observers = []
        self.logger = IndentationDecorator(strategy.model.logger)

    def set_strategy(self, strategy: Classifier) -> None:
        """Allows switching the strategy dynamically."""
        self._strategy = strategy

    def train_model(self, X, y):
        """Trains a model using the classification strategy"""
        self._strategy.train(X, y)

    def evaluate_model(self, X, y) -> float:
        """Evaluates a model using the classification strategy and returns its accuracy"""
        return self._strategy.evaluate(X, y)

    def classify_email(self, email, ts, ic) -> str:
        """Classifies an email using the current strategy."""
        classification = self._strategy.classify(email)
        self._notify_observers(ts, ic, classification)
        return classification

    def save_model(self, file_path):
        """Saves the model in the Classifier"""
        self._strategy.save(file_path)

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

    def _notify_observers(self, ts, ic, classification: str) -> None:
        """Notify observers of a classification."""
        for observer in self._observers:
            observer.update(ts, ic, classification)

    def __str__(self):
        return str(self._strategy)

