from sklearn.neighbors import KNeighborsClassifier
from model.models.base import BaseModel

class KNearestNeighborsModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier()

    def train(self, X, y) -> None:
        print("Training K Nearest Neighbors model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions