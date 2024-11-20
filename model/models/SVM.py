from sklearn.svm import SVC

from model.models.base import BaseModel

class SVMModel(BaseModel):
    def __init__(self) -> None:
        super(SVMModel, self).__init__()
        self.model = SVC(probability=True) # SVM with probability support

    def train(self, X, y) -> None:
        print("Training SVM model...")
        self.model = self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions