from sklearn.neighbors import KNeighborsClassifier
from model.models.base import BaseModel
from utilities.logger.info_logger import InfoLogger

class KNearestNeighborsModel(BaseModel):
    logger = InfoLogger("KNearestNeighborsModel")

    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier()

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions