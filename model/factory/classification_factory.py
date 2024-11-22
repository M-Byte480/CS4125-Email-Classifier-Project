# Factory Pattern - Context Factory
from model.classification_context import ClassificationContext
from model.classifiers.decision_tree_classifier import DecisionTreeClassifier
from model.classifiers.k_nearest_neighbour_classifier import KNearestNeighborsClassifier
from model.classifiers.logistic_regression_classifier import LogisticRegressionClassifier
from model.classifiers.naive_bayes_classifer import NaiveBayesClassifier
from model.classifiers.random_forest_classifier import RandomForestClassifier
from model.classifiers.svm_classifier import SVMClassifier


class ClassificationContextFactory:
    @staticmethod
    def create_context(strategy: str) -> ClassificationContext:
        constructor_selector = {
            "naive_bayes": NaiveBayesClassifier,
            "svm": SVMClassifier,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "logistic_regression": LogisticRegressionClassifier,
            "k_nearest_neighbors": KNearestNeighborsClassifier
        }

        constructor = constructor_selector.get(strategy)

        if constructor:
            return ClassificationContext(constructor())
        else:
            raise ValueError("Invalid strategy")
