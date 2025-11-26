import numpy as np
import pandas as pd
def df_info(df: pd.DataFrame):
    print("\nДискрептивна статистика даних:")
    print(df.describe().T.round(2))
    print("\nДискрептивна статистика позначок часу:")
    ts_temp = pd.to_datetime(df['timestamp'], dayfirst=False, errors='coerce', format='%d.%m.%Y %H:%M:%S')
    print(f"Початок:    {ts_temp.min()}")
    print(f"Кінець:     {ts_temp.max()}")
    print(f"Тривалість: {ts_temp.max() - ts_temp.min()}")
    print(f"Пропусків:  {ts_temp.isna().sum()}\n")


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

def calculate_statistics(data):
    mean = np.mean(data)
    variance = np.var(data)
    std = np.std(data)
    return mean, variance, std

def calculate_process_velocity(y_actual, trend_model, t_points):
    real_velocity = np.diff(y_actual, prepend=y_actual[0])
    
    trend_derivative_func = trend_model.deriv()
    model_velocity = trend_derivative_func(t_points)
    
    return real_velocity, model_velocity