import numpy as np
import pandas as pd

def fit_trend_model(df: pd.DataFrame, degree: int):
    y = df['r_id'].values
    X = np.arange(len(y))
    
    # Метод найменших квадратів
    coeffs = np.polyfit(X, y, degree)
    trend_func = np.poly1d(coeffs)
    y_trend = trend_func(X)
    
    return X, y, y_trend, coeffs

def calculate_residuals(y_actual, y_trend):
    return y_actual - y_trend