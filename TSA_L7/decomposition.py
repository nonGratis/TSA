import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDecomposer:
    """
    Клас для декомпозиції часових рядів.
    Використовує STL (Seasonal-Trend decomposition using LOESS).
    """
    
    def __init__(self, period: Optional[int] = None, seasonal: int = 7, 
                 trend: Optional[int] = None, robust: bool = True):
        """
        Args:
            period: Період сезонності (автовизначення якщо None)
            seasonal: Довжина вікна сезонності (має бути непарним)
            trend: Довжина вікна тренду
            robust: Використовувати робастну декомпозицію
        """
        self.period = period
        self.seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1
        self.trend = trend
        self.robust = robust
        self.result = None
        
    def decompose(self, data: pd.Series) -> Dict[str, pd.Series]:
        """
        Виконує STL декомпозицію часового ряду.
        
        Args:
            data: Часовий ряд (pandas Series з datetime індексом)
            
        Returns:
            Словник з компонентами: trend, seasonal, resid, observed
        """
        # Автовизначення періоду
        if self.period is None:
            self.period = self._auto_detect_period(data)
        
        # Якщо виявлений період забагато великий для довжини даних — зменшуємо до безпечного максимуму
        max_allowed = max(3, len(data) // 2)
        if self.period > max_allowed:
            import warnings
            warnings.warn(
                f"Detected period={self.period} is too large for data length={len(data)}. "
                f"Reducing period to {max_allowed}.", UserWarning
            )
            self.period = max_allowed
        
        # Після корекції перевіряємо чи вистачає даних для STL (потрібно принаймні 2*period)
        if len(data) < 2 * self.period:
            # Для дуже коротких серій повідомляємо інформативну помилку
            raise ValueError(
                f"Недостатньо даних для декомпозиції. Потрібно мінімум {2*self.period} (2*period), є {len(data)}. "
                "Спробуйте вказати менший --decomp-period або використати довший ряд."
            )
        
        # Підбираємо коректний seasonal параметр для STL: непарне, >=3 і < period
        seasonal_param = self.seasonal
        # Якщо seasonal >= period — зменшуємо його до найбільшого допустимого непарного значення
        if seasonal_param >= self.period:
            seasonal_param = self.period - 1
        # Мінімум 3 і робимо непарним
        seasonal_param = max(3, seasonal_param)
        if seasonal_param % 2 == 0:
            seasonal_param += 1
        # На всяк випадок, якщо після корекцій seasonal_param >= period — зменшуємо ще
        if seasonal_param >= self.period:
            seasonal_param = max(3, self.period - 2 if (self.period - 2) >= 3 else 3)
            if seasonal_param % 2 == 0:
                seasonal_param -= 1
        
        # Виконуємо STL декомпозицію з коректованими параметрами
        stl = STL(data, seasonal=seasonal_param, trend=self.trend,
                  period=self.period, robust=self.robust)
        self.result = stl.fit()
        
        return {
            'observed': data,
            'trend': self.result.trend,
            'seasonal': self.result.seasonal,
            'resid': self.result.resid
        }
    
    def _auto_detect_period(self, data: pd.Series) -> int:
        """Автоматичне визначення періоду через ACF."""
        from statsmodels.tsa.stattools import acf
        
        # Обчислюємо ACF
        max_lag = min(len(data) // 2, 100)
        acf_vals = acf(data.dropna(), nlags=max_lag, fft=True)
        
        # Шукаємо перший локальний максимум після lag=0
        for i in range(1, len(acf_vals) - 1):
            if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1]:
                if acf_vals[i] > 0.3:  # Поріг значущості
                    return max(i, 7)  # Мінімум 7
        
        return 24  # За замовчуванням (для годинних даних)
    
    def remove_seasonal(self, data: pd.Series) -> pd.Series:
        """Видаляє сезонну компоненту."""
        if self.result is None:
            self.decompose(data)
        return data - self.result.seasonal
    
    def remove_trend(self, data: pd.Series) -> pd.Series:
        """Видаляє трендову компоненту."""
        if self.result is None:
            self.decompose(data)
        return data - self.result.trend
    
    def get_detrended_deseasonalized(self, data: pd.Series) -> pd.Series:
        """Повертає ряд без тренду і сезонності (тільки залишки)."""
        if self.result is None:
            self.decompose(data)
        return self.result.resid
    
    def get_strength_of_trend(self) -> float:
        """
        Обчислює силу тренду: STL strength.
        Значення близько 1 = сильний тренд, близько 0 = слабкий.
        """
        if self.result is None:
            raise ValueError("Спочатку виконайте декомпозицію")
        
        var_resid = np.var(self.result.resid)
        var_detrend = np.var(self.result.resid + self.result.seasonal)
        
        return max(0, 1 - var_resid / var_detrend) if var_detrend > 0 else 0
    
    def get_strength_of_seasonality(self) -> float:
        """
        Обчислює силу сезонності.
        Значення близько 1 = сильна сезонність, близько 0 = слабка.
        """
        if self.result is None:
            raise ValueError("Спочатку виконайте декомпозицію")
        
        var_resid = np.var(self.result.resid)
        var_deseasonal = np.var(self.result.resid + self.result.trend)
        
        return max(0, 1 - var_resid / var_deseasonal) if var_deseasonal > 0 else 0
    
    def get_statistics(self) -> Dict:
        """Повертає статистику декомпозиції."""
        if self.result is None:
            raise ValueError("Спочатку виконайте декомпозицію")
        
        return {
            'period': self.period,
            'trend_strength': self.get_strength_of_trend(),
            'seasonal_strength': self.get_strength_of_seasonality(),
            'resid_std': float(np.std(self.result.resid)),
            'resid_mean': float(np.mean(self.result.resid)),
            'trend_range': float(self.result.trend.max() - self.result.trend.min()),
            'seasonal_range': float(self.result.seasonal.max() - self.result.seasonal.min())
        }


def quick_decompose(data: pd.Series, period: Optional[int] = None) -> Dict:
    """
    Швидка декомпозиція без створення об'єкту класу.
    
    Args:
        data: Часовий ряд
        period: Період сезонності (автовизначення якщо None)
        
    Returns:
        Словник з компонентами та статистикою
    """
    decomposer = TimeSeriesDecomposer(period=period)
    components = decomposer.decompose(data)
    stats = decomposer.get_statistics()
    
    return {**components, 'statistics': stats}