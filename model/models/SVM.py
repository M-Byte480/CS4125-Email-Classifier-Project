from sklearn.svm import SVC

from model.models.base import BaseModel

class SVMModel(BaseModel):
    def __init__(self) -> None:
        super(SVMModel, self).__init__()
        self.model = SVC(probability=True) # SVM with probability support

    def train(self, X, y) -> None:
        print(f"Training {self.__class__.__name__} model...")
        self.model = self.model.fit(X, y)

    def predict(self, X) -> list:
        print(f"Predicting using {self.__class__.__name__}...")
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "svm"