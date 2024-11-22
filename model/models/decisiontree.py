from sklearn.tree import DecisionTreeClassifier

from model.models.base import BaseModel
from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class DecisionTreeModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionTreeClassifier()
        self.logger = PrefixLogger(self.logger, "DecisionTreeModel")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)#
        return predictions

    def __str__(self):
        return "decision_tree"