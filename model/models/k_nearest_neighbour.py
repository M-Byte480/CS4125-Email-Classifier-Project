from sklearn.neighbors import KNeighborsClassifier
from model.models.base import BaseModel
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class KNearestNeighbourModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier()
        self.logger = PrefixLogger(self.logger, "KNearestNeighborsModel")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions