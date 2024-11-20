from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer

from model.models.base import BaseModel

class NaiveBayesModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.model = MultinomialNB()

    def train(self, X, y) -> None:
        print("Training Naive Bayes model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions
