from sklearn.neighbors import KNeighborsClassifier
from model.models.base import BaseModel

class KNearestNeighborsModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = KNeighborsClassifier()

    def train(self, X, y) -> None:
        print(f"Training {self.__class__.__name__} model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print(f"Predicting using {self.__class__.__name__}...")
        predictions = self.model.predict(X)
        return predictions