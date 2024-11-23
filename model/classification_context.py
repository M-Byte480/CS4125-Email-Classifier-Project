from model.classifier import Classifier

from observers.email_classification_observer import EmailClassificationObserver
from utilities.decorators.timing_decorator import TimingDecorator
from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger

global_timing_decorator = TimingDecorator(InfoLogger())

# Strategy Pattern - Context
class ClassificationContext:
    strategy: Classifier
    _observers: [EmailClassificationObserver]  # Array of subscribed observers

    def __init__(self, strategy: Classifier) -> None:
        self._strategy = strategy
        self._observers = []
        self.info_logger = InfoLogger()
        self.info_logger = PrefixLogger(self.info_logger, "ClassificationContext")
        self.info_logger.log("Classification Context initialized with strategy:" + str(strategy))
        self.timing_decorator = TimingDecorator(self.info_logger)

    def set_strategy(self, strategy: Classifier) -> None:
        """Allows switching the strategy dynamically."""
        self._strategy = strategy

    def train_model(self, X, y):
        """Trains a model using the classification strategy"""
        self._strategy.train(X, y)

    @global_timing_decorator
    def evaluate_model(self, X, y) -> float:
        """Evaluates a model using the classification strategy and returns its accuracy"""
        return self._strategy.evaluate(X, y)

    @global_timing_decorator
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
