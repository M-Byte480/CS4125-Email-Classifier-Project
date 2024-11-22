from sklearn.naive_bayes import MultinomialNB

from model.models.base import BaseModel
from utilities.logger.info_logger import InfoLogger

class NaiveBayesModel(BaseModel):
    logger = InfoLogger("NaiveBayesModel")

    def __init__(self) -> None:
        super().__init__()
        self.model = MultinomialNB()

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "naive_bayes"