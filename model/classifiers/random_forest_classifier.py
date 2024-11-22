from model.classification_context import Classifier
from model.models.randomforest import RandomForest


class RandomForestClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = RandomForest()
