from sklearn.tree import DecisionTreeClassifier

from model.models.base import BaseModel
from utilities.logger.info_logger import InfoLogger


class DecisionTreeModel(BaseModel):
    logger = InfoLogger("DecisionTreeModel")

    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionTreeClassifier()

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)#
        return predictions

    def __str__(self):
        return "decision_tree"