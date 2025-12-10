import numpy as np
from typing import Dict
from statsmodels.tsa.stattools import acf


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обчислення Root Mean Squared Error
    
    Args:
        y_true: Справжні значення
        y_pred: Передбачені значення
        
    Returns:
        RMSE значення
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))


def calculate_bias(residuals: np.ndarray) -> float:
    """
    Обчислення систематичної похибки (bias) - середнє залишків
    
    Args:
        residuals: Масив залишків
        
    Returns:
        Середнє значення залишків
    """
    return float(np.mean(residuals))


def calculate_percent_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обчислення процентної розбіжності
    
    Args:
        y_true: Справжні значення
        y_pred: Передбачені значення
        
    Returns:
        Середня процентна розбіжність (%)
    """
    # Уникаємо ділення на нуль
    mask = np.abs(y_true) > 1e-10
    
    if not mask.any():
        return 0.0
    
    percent_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0
    return float(np.mean(percent_errors))


def calculate_acf(residuals: np.ndarray, nlags: int = 40) -> np.ndarray:
    """
    Обчислення автокореляційної функції (ACF) залишків
    
    Args:
        residuals: Масив залишків
        nlags: Кількість лагів для обчислення
        
    Returns:
        Масив значень ACF
    """
    # Обмежуємо nlags довжиною ряду
    nlags = min(nlags, len(residuals) - 1)
    
    if nlags < 1:
        return np.array([1.0])
    
    acf_values = acf(residuals, nlags=nlags, fft=True)
    return acf_values


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обчислення Mean Absolute Error
    
    Args:
        y_true: Справжні значення
        y_pred: Передбачені значення
        
    Returns:
        MAE значення
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Обчислення коефіцієнта детермінації R²
    
    Args:
        y_true: Справжні значення
        y_pred: Передбачені значення
        
    Returns:
        R² значення
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    return float(1.0 - (ss_res / ss_tot))


def evaluate_filter_performance(
    y_true: np.ndarray,
    y_filtered: np.ndarray,
    residuals: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Комплексна оцінка якості фільтрації
    
    Args:
        y_true: Справжні значення (імпутовані)
        y_filtered: Фільтровані значення (Kalman)
        residuals: Залишки
        verbose: Виводити результати
        
    Returns:
        Словник з метриками:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - r2: Коефіцієнт детермінації
            - bias: Систематична похибка
            - percent_divergence: Процентна розбіжність
            - residual_std: Стандартне відхилення залишків
            - residual_mean: Середнє залишків
            - acf_lag1: Автокореляція на лазі 1
    """
    metrics = {
        'rmse': calculate_rmse(y_true, y_filtered),
        'mae': calculate_mae(y_true, y_filtered),
        'r2': calculate_r2(y_true, y_filtered),
        'bias': calculate_bias(residuals),
        'percent_divergence': calculate_percent_divergence(y_true, y_filtered),
        'residual_std': float(np.std(residuals, ddof=1)),
        'residual_mean': float(np.mean(residuals))
    }
    
    # ACF на лазі 1 (якщо достатньо даних)
    if len(residuals) > 2:
        acf_values = calculate_acf(residuals, nlags=1)
        metrics['acf_lag1'] = float(acf_values[1]) if len(acf_values) > 1 else 0.0
    else:
        metrics['acf_lag1'] = 0.0
    
    if verbose:
        print("МЕТРИКИ ЯКОСТІ ФІЛЬТРАЦІЇ")
        print(f"RMSE:                  {metrics['rmse']:.4f}")
        print(f"MAE:                   {metrics['mae']:.4f}")
        print(f"R²:                    {metrics['r2']:.4f}")
        print(f"Bias (середнє залишків): {metrics['bias']:.4f}")
        print(f"% розбіжності:         {metrics['percent_divergence']:.2f}%")
        print(f"Std залишків:          {metrics['residual_std']:.4f}")
        print(f"ACF(1):                {metrics['acf_lag1']:.4f}")
    
    return metrics


def check_residuals_whiteness(residuals: np.ndarray, nlags: int = 40, alpha: float = 0.05) -> Dict:
    """
    Перевірка залишків на "білий шум" через ACF
    
    Args:
        residuals: Масив залишків
        nlags: Кількість лагів для перевірки
        alpha: Рівень значущості для довірчих меж
        
    Returns:
        Словник з результатами:
            - acf_values: Значення ACF
            - significant_lags: Індекси лагів зі значущою кореляцією
            - is_white_noise: Чи є залишки білим шумом
    """
    acf_values = calculate_acf(residuals, nlags=nlags)
    
    # Довірчі межі для білого шуму: ±1.96/√n
    n = len(residuals)
    confidence_bound = 1.96 / np.sqrt(n)
    
    # Знаходимо значущі лаги (крім lag=0, який завжди = 1)
    significant_lags = np.where(np.abs(acf_values[1:]) > confidence_bound)[0] + 1
    
    # Білий шум, якщо немає значущих кореляцій
    is_white_noise = len(significant_lags) == 0
    
    return {
        'acf_values': acf_values,
        'significant_lags': significant_lags.tolist(),
        'is_white_noise': is_white_noise,
        'confidence_bound': confidence_bound
    }
