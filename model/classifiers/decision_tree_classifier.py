from model.classification_context import Classifier
from model.models.decisiontree import DecisionTreeModel


class DecisionTreeClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeModel()