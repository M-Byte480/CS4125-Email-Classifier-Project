from abc import ABC, abstractmethod
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
        return "spam"

class SVMClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with SVM")
        return "not spam"

class DecisionTreeClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with Decision Tree")
        return "spam"

class RandomForestClassifier(ClassificationStrategy):
    def classify(self, email: Email):
        # Logic
        print("Classifying with Random Forest")
        return "not spam"

class ClassificationContext:
    strategy: ClassificationStrategy

    def __init__(self, strategy: ClassificationStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ClassificationStrategy) -> None:
        """Allows switching the strategy dynamically."""
        self._strategy = strategy

    def classify_email(self, email: Email) -> str:
        """Classifies an email using the current strategy."""
        return self._strategy.classify(email)