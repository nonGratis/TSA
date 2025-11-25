import numpy as np
import pandas as pd

def fit_trend_model(df: pd.DataFrame, degree: int):
    y = np.array(df['r_id'].values, dtype=float)
    X = np.arange(len(y))
    
    # Метод найменших квадратів
    coeffs = np.polyfit(X, y, degree)
    trend_func = np.poly1d(coeffs)
    y_trend = trend_func(X)
    
    return X, y, y_trend, coeffs, trend_func

def calculate_residuals(y_actual, y_trend):
    return y_actual - y_trend

def calculate_statistics(residuals):
    mean = np.mean(residuals)
    variance = np.var(residuals)
    std = np.std(residuals)
    
    return mean, variance, std

def calculate_process_velocity(y_actual, trend_model, t_points):
    real_velocity = np.diff(y_actual, prepend=y_actual[0])
    
    trend_derivative_func = trend_model.deriv()
    model_velocity = trend_derivative_func(t_points)
    
    return real_velocity, model_velocity