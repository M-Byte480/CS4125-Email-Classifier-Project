from abc import ABC, abstractmethod

from observers.email_classification_observer import EmailClassificationObserver
from structs.objects import Email

# Strategy Pattern
class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, email: Email):
        pass

class NaiveBayesClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with Naive Bayes")
        return email.classification

class SVMClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with SVM")
        return email.classification

class DecisionTreeClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with Decision Tree")
        return email.classification

class RandomForestClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with Random Forest")
        return email.classification

# Strategy Pattern
# ClassificationContext is also a Subject that holds a list of Observers
class ClassificationContext:
    _strategy: ClassificationStrategy
    _observers: [EmailClassificationObserver] # Array of subscribed observers

    def __init__(self, strategy: ClassificationStrategy) -> None:
        self._strategy = strategy
        self._observers = []

    def set_strategy(self, strategy: ClassificationStrategy) -> None:
        """Allows switching the strategy dynamically."""
        self._strategy = strategy

    def classify_email(self, email: Email) -> str:
        """Classifies an email using the current strategy and notify observers."""
        classification: str = self._strategy.classify(email)
        self._notify_observers(classification)
        return classification

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

# Factory Pattern
class ClassificationContextFactory:
    @staticmethod
    def create_context(strategy: str) -> ClassificationContext:
        match strategy:
            case "naive_bayes":
                return ClassificationContext(NaiveBayesClassifier())
            case "svm":
                return ClassificationContext(SVMClassifier())
            case "decision_tree":
                return ClassificationContext(DecisionTreeClassifier())
            case "random_forest":
                return ClassificationContext(RandomForestClassifier())
            case _:
                raise ValueError("Invalid strategy")