import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class RegressionForecaster:
    def __init__(self):
        self.models = {}

    def _prepare_features(self, data_len, steps):
        # X - це просто індекси часу: [0, 1, 2, ..., N]
        X_train = np.arange(data_len).reshape(-1, 1)
        X_future = np.arange(data_len, data_len + steps).reshape(-1, 1)
        return X_train, X_future

    def linear_regression(self, data: pd.Series, steps: int):
        X_train, X_future = self._prepare_features(len(data), steps)
        y_train = data.values

        model = LinearRegression()
        model.fit(X_train, y_train)
        
        forecast = model.predict(X_future)
        self.models['linear'] = model
        return forecast

    def polynomial_regression(self, data: pd.Series, steps: int, degree: int = 2):
        X_train, X_future = self._prepare_features(len(data), steps)
        y_train = data.values

        # Створюємо пайплайн: Поліноміальні ознаки -> Лінійна регресія
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        
        forecast = model.predict(X_future)
        self.models[f'poly_{degree}'] = model
        return forecast