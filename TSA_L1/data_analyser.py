import numpy as np
import pandas as pd
from scipy import stats
from data_handler import check_timestamp

def df_info(df: pd.DataFrame):
    df = check_timestamp(df)
    
    print(f"Кількість записів: {len(df)}")
    
    ts = df['timestamp']
    
    print(f"\nЧасовий діапазон:")
    print(f"    Початок:    {ts.min()}")
    print(f"    Кінець:     {ts.max()}")
    print(f"    Тривалість: {ts.max() - ts.min()}")
    
    print(f"\nСтатистика по r_id:")
    r_min = df['r_id'].min()
    r_max = df['r_id'].max()
    r_mean = df['r_id'].mean()
    r_std = df['r_id'].std()
    
    print(f"    Min:  {r_min:.2f}" if pd.notna(r_min) else "    Min:  NaN")
    print(f"    Max:  {r_max:.2f}" if pd.notna(r_max) else "    Max:  NaN")
    print(f"    Mean: {r_mean:.2f}" if pd.notna(r_mean) else "    Mean: NaN")
    print(f"    Std:  {r_std:.2f}" if pd.notna(r_std) else "    Std:  NaN")

def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        print(f"\nseed: {seed}")
    else:
        print("\nseed: random")


def calculate_residuals(y_actual, y_trend):
    return y_actual - y_trend


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

def print_statistics_report(y, y_trend, y_synthetic, residuals, residuals_synthetic, coeffs, distribution):
    """
    Виведення повного статистичного звіту:
    - Коефіцієнти моделі
    - Тест нормальності залишків
    - Таблиця статистики для всіх компонентів
    - Тест Колмогорова-Смірнова
    """
    print(f"\nКоєфіцієнти моделі (старший-молодший): {coeffs}")
    
    
    print(f"Тип розподілу шуму: {distribution}")
    
    p_value = check_normality(residuals)
    print(f"\nТест нормальності (Шапіро-Вілка) залишків теоретичної моделі, p-value: {p_value:.4e}")
    
    print(f"\n{'Компонента':<30} | {'M (μ)':<12} | {'D (σ²)':<12} | {'Std (σ)':<12}")
    print("-" * 75)

    datasets = {
        "Експерементальні дані": y,
        "Теоретична модель": y_trend,
        "Залишки теоретичної моделі": residuals,
        "Синтетична модель": y_synthetic,
        "Залишки синтетичної моделі": residuals_synthetic
    }
    for name, data in datasets.items():
        m, v, s = calculate_statistics(data)
        print(f"{name:<30} | {m:<12.2f} | {v:<12.2f} | {s:<12.2f}")
    print("-" * 75)
    
    ks_stat, ks_p_value = compare_distributions_ks(residuals, residuals_synthetic)
    print("\nТест Колмогорова-Смірнова (порівняння розподілів реального та синтетичного шумів):")
    print(f"Statistic: {ks_stat:.4f}, p-value: {ks_p_value:.4f}")