import numpy as np
import pandas as pd
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesProperties:
    """
    Клас для аналізу властивостей часових рядів:
    - Стаціонарність (ADF, KPSS тести)
    - Фрактальний аналіз (Hurst exponent)
    - Автокореляція (ACF/PACF)
    """
    
    def __init__(self):
        self.properties = {}
    
    def test_stationarity(self, data: pd.Series, alpha: float = 0.05) -> Dict:
        """
        Тестування стаціонарності через ADF та KPSS тести.
        
        Args:
            data: Часовий ряд
            alpha: Рівень значущості (за замовчуванням 5%)
            
        Returns:
            Словник з результатами обох тестів
        """
        clean_data = data.dropna()
        
        # ADF test (Augmented Dickey-Fuller)
        # H0: ряд НЕстаціонарний (має одиничний корінь)
        # Якщо p-value < alpha → відхиляємо H0 → ряд стаціонарний
        adf_result = adfuller(clean_data, autolag='AIC')
        
        adf_dict = {
            'test_statistic': float(adf_result[0]),
            'p_value': float(adf_result[1]),
            'n_lags': int(adf_result[2]),
            'n_obs': int(adf_result[3]),
            'critical_values': {k: float(v) for k, v in adf_result[4].items()},
            'is_stationary': adf_result[1] < alpha
        }
        
        # KPSS test (Kwiatkowski-Phillips-Schmidt-Shin)
        # H0: ряд стаціонарний
        # Якщо p-value < alpha → відхиляємо H0 → ряд НЕстаціонарний
        try:
            kpss_result = kpss(clean_data, regression='c', nlags='auto')
            
            kpss_dict = {
                'test_statistic': float(kpss_result[0]),
                'p_value': float(kpss_result[1]),
                'n_lags': int(kpss_result[2]),
                'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                'is_stationary': kpss_result[1] >= alpha
            }
        except Exception as e:
            kpss_dict = {'error': str(e), 'is_stationary': None}
        
        # Загальний висновок
        if adf_dict['is_stationary'] and kpss_dict.get('is_stationary', False):
            conclusion = 'stationary'
        elif not adf_dict['is_stationary'] and not kpss_dict.get('is_stationary', True):
            conclusion = 'non_stationary'
        else:
            conclusion = 'difference_stationary'  # Потребує диференціювання
        
        return {
            'adf': adf_dict,
            'kpss': kpss_dict,
            'conclusion': conclusion
        }
    
    def calculate_hurst_exponent(self, data: pd.Series) -> Dict:
        """
        Обчислення показника Hurst (фрактальний аналіз).
        
        H < 0.5: Антиперсистентний (mean-reverting) - тренд схильний змінюватись
        H = 0.5: Випадкове блукання (Brownian motion)
        H > 0.5: Персистентний (trending) - тренд схильний продовжуватись
        
        Args:
            data: Часовий ряд
            
        Returns:
            Словник з H та інтерпретацією
        """
        clean_data = data.dropna().values
        
        if len(clean_data) < 100:
            return {'hurst': None, 'error': 'Недостатньо даних (потрібно мінімум 100)'}
        
        # R/S аналіз (Rescaled Range)
        lags = self._get_hurst_lags(len(clean_data))
        rs_values = []
        
        for lag in lags:
            n_chunks = len(clean_data) // lag
            if n_chunks < 2:
                continue
                
            rs_chunk = []
            for i in range(n_chunks):
                chunk = clean_data[i*lag:(i+1)*lag]
                
                # Середнє центроване кумулятивне відхилення
                mean_chunk = np.mean(chunk)
                y = np.cumsum(chunk - mean_chunk)
                
                # Range
                R = np.max(y) - np.min(y)
                
                # Standard deviation
                S = np.std(chunk, ddof=1)
                
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if len(rs_values) < 3:
            return {'hurst': None, 'error': 'Недостатньо точок для оцінки'}
        
        # Лінійна регресія log(R/S) = H * log(n) + const
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Видаляємо inf та nan
        mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if mask.sum() < 3:
            return {'hurst': None, 'error': 'Недостатньо валідних точок'}
        
        # МНК
        H = np.polyfit(log_lags[mask], log_rs[mask], 1)[0]
        
        # Інтерпретація
        if H < 0.45:
            interpretation = 'anti_persistent'
            description = 'Процес схильний до зміни напрямку (mean-reverting)'
        elif H > 0.55:
            interpretation = 'persistent'
            description = 'Процес схильний продовжувати поточний тренд'
        else:
            interpretation = 'random_walk'
            description = 'Випадкове блукання (Brownian motion)'
        
        return {
            'hurst': float(H),
            'interpretation': interpretation,
            'description': description
        }
    
    def _get_hurst_lags(self, n: int) -> np.ndarray:
        """Генерує лаги для R/S аналізу."""
        min_lag = 8
        max_lag = n // 4
        n_lags = min(20, max_lag // min_lag)
        return np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), 
                                     n_lags).astype(int))
    
    def calculate_autocorrelation(self, data: pd.Series, 
                                  nlags: int = 40) -> Dict:
        """
        Обчислення ACF та PACF.
        
        Args:
            data: Часовий ряд
            nlags: Кількість лагів
            
        Returns:
            Словник з ACF, PACF та довірчими межами
        """
        clean_data = data.dropna()
        nlags = min(nlags, len(clean_data) // 2 - 1)
        
        # ACF
        acf_vals = acf(clean_data, nlags=nlags, fft=True)
        
        # PACF
        try:
            pacf_vals = pacf(clean_data, nlags=nlags, method='ywm')
        except:
            pacf_vals = np.full(nlags + 1, np.nan)
        
        # Довірчі межі (95%)
        conf_bound = 1.96 / np.sqrt(len(clean_data))
        
        # Значущі лаги
        significant_acf = np.where(np.abs(acf_vals[1:]) > conf_bound)[0] + 1
        significant_pacf = np.where(np.abs(pacf_vals[1:]) > conf_bound)[0] + 1
        
        return {
            'acf': acf_vals,
            'pacf': pacf_vals,
            'confidence_bound': float(conf_bound),
            'significant_acf_lags': significant_acf.tolist(),
            'significant_pacf_lags': significant_pacf.tolist(),
            'n_significant_acf': len(significant_acf),
            'n_significant_pacf': len(significant_pacf)
        }
    
    def analyze_all(self, data: pd.Series, nlags: int = 40) -> Dict:
        """
        Комплексний аналіз всіх властивостей.
        
        Args:
            data: Часовий ряд
            nlags: Кількість лагів для ACF/PACF
            
        Returns:
            Словник з усіма результатами
        """
        print("\n=== АНАЛІЗ ВЛАСТИВОСТЕЙ ЧАСОВОГО РЯДУ ===\n")
        
        # Стаціонарність
        print("[1/3] Тестування стаціонарності...")
        stationarity = self.test_stationarity(data)
        
        print(f"  ADF тест: p-value = {stationarity['adf']['p_value']:.4f}")
        print(f"  → Ряд {'СТАЦІОНАРНИЙ' if stationarity['adf']['is_stationary'] else 'НЕ стаціонарний'} (ADF)")
        
        if 'error' not in stationarity['kpss']:
            print(f"  KPSS тест: p-value = {stationarity['kpss']['p_value']:.4f}")
            print(f"  → Ряд {'СТАЦІОНАРНИЙ' if stationarity['kpss']['is_stationary'] else 'НЕ стаціонарний'} (KPSS)")
        
        print(f"\n  Висновок: {stationarity['conclusion'].upper()}")
        
        # Hurst
        print("\n[2/3] Фрактальний аналіз (Hurst exponent)...")
        hurst = self.calculate_hurst_exponent(data)
        
        if hurst['hurst'] is not None:
            print(f"  H = {hurst['hurst']:.4f}")
            print(f"  → {hurst['description']}")
        else:
            print(f"  Помилка: {hurst.get('error', 'Unknown')}")
        
        # ACF/PACF
        print("\n[3/3] Автокореляційний аналіз...")
        autocorr = self.calculate_autocorrelation(data, nlags)
        
        print(f"  Значущих ACF лагів: {autocorr['n_significant_acf']}")
        print(f"  Значущих PACF лагів: {autocorr['n_significant_pacf']}")
        
        if autocorr['n_significant_acf'] > 0:
            print(f"  Перші 5 значущих ACF лагів: {autocorr['significant_acf_lags'][:5]}")
        
        print("\n" + "="*50 + "\n")
        
        return {
            'stationarity': stationarity,
            'hurst': hurst,
            'autocorrelation': autocorr,
            'summary': {
                'is_stationary': stationarity['conclusion'] == 'stationary',
                'hurst_value': hurst.get('hurst'),
                'has_autocorrelation': autocorr['n_significant_acf'] > 0
            }
        }


def quick_properties_check(data: pd.Series) -> Dict:
    """Швидка перевірка основних властивостей."""
    analyzer = TimeSeriesProperties()
    return analyzer.analyze_all(data)