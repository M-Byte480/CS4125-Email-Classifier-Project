from sklearn.naive_bayes import MultinomialNB

from model.models.base import BaseModel
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class NaiveBayesModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = MultinomialNB()
        self.logger = PrefixLogger(self.logger, "NaiveBayesModel")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "naive_bayes"