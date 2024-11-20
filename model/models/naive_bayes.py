from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer

from model.models.base import BaseModel


class NaiveBayesModel(BaseModel):


    def __init__(self) -> None:
        super().__init__()
        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        self.label_binarizer = LabelBinarizer()

    def train(self, X, y) -> None:
        # self.fit_vectorizer(X)

        # print("Transforming data...")
        # X_transformed = self.data_transform(X)
        # y_transformed = self.label_binarizer.fit_transform(y)

        # print("Training Naive Bayes model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        # print("Transforming data for prediction...")
        # X_transformed = self.data_transform(X)

        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions
        # return self.label_binarizer.inverse_transform(predictions)

    def data_transform(self, X) -> any:
        return self.vectorizer.transform(X)

    def fit_vectorizer(self, X):
        print("Fitting vectorizer...")
        self.vectorizer.fit(X)
