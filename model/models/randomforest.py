import random

from sklearn.ensemble import RandomForestClassifier

from model.models.base import BaseModel
from utilities.logger.decorators.prefix_decorator import PrefixLogger

class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        seed = random.randint(1, 1000)
        self.model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.logger = PrefixLogger(self.logger, "LogisticRegressionModel")

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> list:
        predictions = self.model.predict(X)
        return predictions

    def __str__(self):
        return "random_forest"