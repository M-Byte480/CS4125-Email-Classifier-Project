import random

from sklearn.ensemble import RandomForestClassifier

from model.models.base import BaseModel
from utilities.logger.info_logger import InfoLogger

seed = random.randint(1, 1000)

class RandomForest(BaseModel):
    logger = InfoLogger("RandomForest")

    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "random_forest"