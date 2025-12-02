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

def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        print(f"seed: {seed}")
    else:
        print("seed: random")

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

def calculate_r_squared(y_actual, y_trend):
    ss_res = np.sum((y_actual - y_trend) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        print("ss_tot = 0. R² не визначений (повертається np.nan), всі вхідні знач. однакові")
        r_squared = np.nan
    else:
        r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared

def calculate_adjusted_r_squared(r_squared, n, num_params):
    if n - num_params <= 0:
        return 0.0
    
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - num_params))
    return adj_r_squared

def calculate_f_statistic(r_squared, num_params, n):
    k = num_params - 1
    
    if k > 0 and (n - k - 1) > 0 and r_squared < 1.0:
        f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
        p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    else:
        f_stat = 0.0
        p_value = 1.0
    
    return f_stat, p_value

def calculate_statistics(data):
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    std = np.std(data, ddof=1)
    return mean, variance, std

def check_normality(residuals):
    stat, p_value = stats.shapiro(residuals)
    return p_value

def compare_distributions_ks(data1, data2):
    statistic, p_value = stats.ks_2samp(data1, data2) # Колмогорова-Смірнова для двох вибірок
    return statistic, p_value

def generate_noise(std, size, distribution='normal'):
    if distribution == 'normal':
        noise = np.random.normal(loc=0, scale=std, size=size)
    elif distribution == 'uniform':
        delta = std * np.sqrt(3)
        noise = np.random.uniform(low=-delta, high=delta, size=size)
    elif distribution == 'exponential':
        noise = np.random.exponential(scale=std, size=size)
        noise = noise - np.mean(noise)
        
    else:
        raise ValueError(f"Невідомий тип розподілу: {distribution}")
    
    return noise

def generate_synthetic_data(y_trend, std, distribution='normal'):
    noise = generate_noise(std, len(y_trend), distribution)
    return y_trend + noise