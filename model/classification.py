from abc import ABC, abstractmethod

from scipy.stats import studentized_range_gen

from model.models.SVM import SVMModel
from model.models.decisiontree import DecisionTreeModel
from model.models.logistic_regression import LogisticRegressionModel
from model.models.naive_bayes import NaiveBayesModel
from model.models.randomforest import RandomForest
from structs.objects import Email

# Strategy Pattern
class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, email: Email):
        pass

class NaiveBayesClassifier(ClassificationStrategy):
    def __init__(self):
        self.model = NaiveBayesModel()

    def classify(self, email):
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

class SVMClassifier(ClassificationStrategy):
    def __init__(self):
        self.model = SVMModel()

    def classify(self, email: Email):
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

class DecisionTreeClassifier(ClassificationStrategy):
    def __init__(self):
        self.model = DecisionTreeModel()

    def classify(self, email):
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

class RandomForestClassifier(ClassificationStrategy):
    def __init__(self):
        self.model = RandomForest()

    def classify(self, email: Email):
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

class LogisticRegressionClassifier(ClassificationStrategy):
    def __init__(self):
        self.model = LogisticRegressionModel()

    def classify(self, email):
        print(f"Classifying email: {email}")
        prediction = self.model.predict([email])
        return prediction

# Strategy Pattern
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

# Factory Pattern
class ClassificationContextFactory:
    @staticmethod
    def create_context(strategy: str) -> ClassificationContext:
        function_list = {
            "naive_bayes": NaiveBayesClassifier,
            "svm": SVMClassifier,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegressionClassifier
        }

        f = function_list.get(strategy)

        if f:
            return ClassificationContext(f())
        else:
            raise ValueError("Invalid strategy")
