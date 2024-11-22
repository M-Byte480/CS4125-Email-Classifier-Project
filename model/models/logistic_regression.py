from sklearn.linear_model import LogisticRegression

from model.models.base import BaseModel
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class LogisticRegressionModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=1000)
        self.logger = PrefixLogger(self.logger, "LogisticRegressionModel")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "logistic_regression"