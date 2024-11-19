from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

from model.models.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from numpy import *


class SVMModel(BaseModel):

    __slots__ = ["model_name", "embeddings", "y", "model", "vectorizer", "label_binarizer"]

    def __init__(self) -> None:
        super(SVMModel, self).__init__()
        self.model = SVC(probability=True)          # SVM with probability support
        self.model_name = "SVM"
        self.vectorizer = TfidfVectorizer()         # Text to numeric transformation
        self.label_binarizer = LabelBinarizer()     # For multi-class labels

    # @Override
    def data_transform(self, Z) -> any:
        print("Transforming data...")
        return self.vectorizer.transform(Z)

    def fit_vectorizer(self, X):
        print("Fitting vectorizer...")
        self.vectorizer.fit(X)

    # @Override
    def train(self, X, y) -> None:
        print("Transforming data...")
        X = self.data_transform(X)
        y = self.label_binarizer.fit_transform(y)

        print("Training SVM model...")
        self.model = self.model.fit(X, y)

    # @Overload
    def train(self, data) -> None:
        self.train(data.X_train, data.y_train)


    def predict(self, X) -> any:
        print("Transforming data for prediction...")
        X_transformed = self.data_transform(X)

        print("Predicting...")
        predictions = self.model.predict(X_transformed)
        return self.label_binarizer.inverse_transform(predictions)