from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from model.models.base import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = DecisionTreeClassifier()  
        self.vectorizer = TfidfVectorizer() 
        self.label_binarizer = LabelBinarizer()  

    def train(self, X, y) -> None:
        print("Transforming data...")
        X_transformed = self.data_transform(X)
        y_transformed = self.label_binarizer.fit_transform(y)

        print("Training Decision Tree model...")
        self.model.fit(X_transformed, y_transformed)

    def train(self, data) -> None:
        self.train(data.X_train, data.y_train)

    def predict(self, X) -> list:

        print("Transforming data for prediction...")
        X_transformed = self.data_transform(X)

        print("Predicting...")
        predictions = self.model.predict(X_transformed)
        return self.label_binarizer.inverse_transform(predictions)

    def data_transform(self, X) -> any:
        return self.vectorizer.transform(X)

    def fit_vectorizer(self, X):
        print("Fitting vectorizer...")
        self.vectorizer.fit(X)