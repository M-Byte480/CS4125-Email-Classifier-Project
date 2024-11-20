from sklearn.linear_model import LogisticRegression

from model.models.base import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y) -> None:
        print("Training Logistic Regression model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions
