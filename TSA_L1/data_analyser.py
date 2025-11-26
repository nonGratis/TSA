import numpy as np
import pandas as pd
from scipy import stats

def df_info(df: pd.DataFrame):
    print(f"Кількість записів: {len(df)}")
    
    ts = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')
    
    print(f"\nЧасовий діапазон:")
    print(f"    Початок:    {ts.min()}")
    print(f"    Кінець:     {ts.max()}")
    print(f"    Тривалість: {ts.max() - ts.min()}")
    
    print(f"\nСтатистика по r_id:")
    print(f"    Min:  {df['r_id'].min():.2f}")
    print(f"    Max:  {df['r_id'].max():.2f}")
    print(f"    Mean: {df['r_id'].mean():.2f}")
    print(f"    Std:  {df['r_id'].std():.2f}")


def fit_polynomial_trend(df: pd.DataFrame, degree: int):
    y = np.array(df['r_id'].values, dtype=float)
    X = np.arange(len(y))
    
    # Метод найменших квадратів
    coeffs = np.polyfit(X, y, degree)
    trend_func = np.poly1d(coeffs)
    y_trend = trend_func(X)
    
    return X, y, y_trend, coeffs

def fit_logarithmic_trend(df: pd.DataFrame):
    y = np.array(df['r_id'].values, dtype=float)
    X = np.arange(1, len(y) + 1)
    
    X_log = np.log(X)
    coeffs = np.polyfit(X_log, y, 1)
    
    y_trend = coeffs[0] * X_log + coeffs[1]
    
    return np.arange(len(y)), y, y_trend, coeffs

def calculate_residuals(y_actual, y_trend):
    return y_actual - y_trend

def calculate_statistics(data):
    mean = np.mean(data)
    variance = np.var(data)
    std = np.std(data)
    return mean, variance, std

def check_normality(residuals):
    stat, p_value = stats.shapiro(residuals)
    return p_value
