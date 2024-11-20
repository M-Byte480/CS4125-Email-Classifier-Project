from sklearn.tree import DecisionTreeClassifier

from model.models.base import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionTreeClassifier()

    def train(self, X, y) -> None:
        print("Training Decision Tree model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)#
        return predictions