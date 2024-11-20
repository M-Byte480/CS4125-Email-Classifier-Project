import random

from sklearn.ensemble import RandomForestClassifier

from model.models.base import BaseModel

seed = random.randint(1, 1000)

class RandomForest(BaseModel):
    def __init__(self) -> None:
        super(RandomForest, self).__init__()
        self.model = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')

    def train(self, X, y) -> None:
        print("Training Random Forest model...")
        self.model.fit(X, y)

    def predict(self, X) -> list:
        print("Predicting...")
        predictions = self.model.predict(X)
        return predictions
