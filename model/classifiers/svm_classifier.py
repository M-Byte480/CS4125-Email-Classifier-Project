from model.classification_context import Classifier
from model.models.SVM import SVMModel


class SVMClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = SVMModel()