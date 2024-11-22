# Strategy Pattern - Concrete Classification Classes
from model.classification_context import Classifier
from model.models.naive_bayes import NaiveBayesModel


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = NaiveBayesModel()