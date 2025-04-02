import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

class LightGBMPredictor:
    def __init__(self, task="classification", params=None):
        self.task = task
        default_params = {"learning_rate": 0.05, "n_estimators": 100}
        self.params = {**default_params, **(params or {})}
        self.model = None

    def fit(self, X, y):
        if self.task == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        preds = self.model.predict(X)
        if self.task == "classification":
            preds = (preds > 0.5).astype(int)
        return preds

    def evaluate(self, X, y):
        preds = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, preds)
        else:
            return np.sqrt(mean_squared_error(y, preds))
