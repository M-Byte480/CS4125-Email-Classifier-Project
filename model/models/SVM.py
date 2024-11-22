from sklearn.svm import SVC

from model.models.base import BaseModel
from utilities.logger.info_logger import InfoLogger

class SVMModel(BaseModel):
    logger = InfoLogger("SVMModel")

    def __init__(self) -> None:
        super(SVMModel, self).__init__()
        self.model = SVC(probability=True) # SVM with probability support

    def train(self, X, y) -> None:
        self.model = self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "svm"