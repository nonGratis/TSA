import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any

class ModelSynthesizer:
    """
    Автоматичний синтез математичної моделі часового ряду
    відповідно до лекційних матеріалів (термін №3-5, №35)
    """
    
    def __init__(self, y: np.ndarray, max_degree: int = 6):
        """
        Args:
            y: Вхідний часовий ряд
            max_degree: Максимальний порядок полінома для перевірки
        """
        self.y = y
        self.n = len(y)
        self.X = np.arange(self.n)
        self.max_degree = min(max_degree, self.n // 2 - 1)
        
    def analyze_trend_type(self) -> Dict[str, Any]:
        """
        Етап 1: Визначення характеру тренду
        
        Returns:
            dict: {
                'type': 'monotonic' | 'seasonal' | 'mixed',
                'monotonic_score': float,
                'seasonality_detected': bool
            }
        """
        dy = np.diff(self.y)
        
        sign_changes = np.sum(np.diff(np.sign(dy)) != 0)
        monotonic_score = 1.0 - (sign_changes / len(dy))
        
        seasonality_detected = False
        if self.n > 20:
            fft = np.fft.fft(self.y - np.mean(self.y))
            power = np.abs(fft[:self.n//2])**2
            if len(power) > 5:
                mean_power = np.mean(power[2:])
                max_power = np.max(power[2:])
                if max_power > 3 * mean_power:
                    seasonality_detected = True
        
        if seasonality_detected:
            trend_type = 'seasonal'
        elif monotonic_score > 0.7:
            trend_type = 'monotonic'
        else:
            trend_type = 'mixed'
            
        return {
            'type': trend_type,
            'monotonic_score': monotonic_score,
            'seasonality_detected': seasonality_detected
        }
    
    def select_model_class(self, trend_info: Dict) -> str:
        if trend_info['seasonality_detected']:
            return 'poly'        
        if trend_info['monotonic_score'] > 0.8:
            if self._test_logarithmic():
                return 'log'
            elif self._test_exponential():
                return 'exp'        
        return 'poly'
    
    def _test_logarithmic(self) -> bool:
        """Перевірка відповідності логарифмічній моделі"""
        if self.n < 10:
            return False
        
        X_log = np.log(self.X[1:] + 1)
        y_subset = self.y[1:]
        
        coeffs = np.polyfit(X_log, y_subset, 1)
        y_pred = coeffs[0] * X_log + coeffs[1]
        
        ss_res = np.sum((y_subset - y_pred)**2)
        ss_tot = np.sum((y_subset - np.mean(y_subset))**2)
        r2_log = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r2_log > 0.85
    
    def _test_exponential(self) -> bool:
        """Перевірка відповідності експоненційній моделі"""
        if self.n < 10 or np.any(self.y <= 0):
            return False
        
        y_log = np.log(self.y)
        coeffs = np.polyfit(self.X, y_log, 1)
        y_pred = coeffs[0] * self.X + coeffs[1]
        
        ss_res = np.sum((y_log - y_pred)**2)
        ss_tot = np.sum((y_log - np.mean(y_log))**2)
        r2_exp = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r2_exp > 0.85    

    def build_trend(self, model_type: str, degree: Optional[int] = None, coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Побудова тренду для заданого типу моделі
        
        Args:
            model_type: 'poly' | 'log' | 'exp'
            degree: Порядок полінома (для poly)
            coeffs: Коефіцієнти моделі (якщо вже обчислені)
            
        Returns:
            (y_trend, coeffs, model_type): Тренд, коефіцієнти та тип моделі для візуалізації
        """
        if model_type == 'poly':
            if coeffs is None:
                if degree is None:
                    raise ValueError("Для poly моделі потрібен degree або coeffs")
                coeffs = np.polyfit(self.X, self.y, degree)
            y_trend = np.poly1d(coeffs)(self.X)
            
        elif model_type == 'log':
            if coeffs is None:
                X_log = np.log(self.X[1:] + 1)
                coeffs = np.polyfit(X_log, self.y[1:], 1)
            
            y_trend = np.zeros_like(self.y, dtype=float)
            y_trend[0] = coeffs[1]
            X_log = np.log(self.X[1:] + 1)
            y_trend[1:] = coeffs[0] * X_log + coeffs[1]
            
        elif model_type == 'exp':
            if coeffs is None:
                y_log = np.log(self.y)
                coeffs_log = np.polyfit(self.X, y_log, 1)
                coeffs = np.array([np.exp(coeffs_log[1]), coeffs_log[0]])
            
            y_trend = coeffs[0] * np.exp(coeffs[1] * self.X)
            
        else:
            raise ValueError(f"Невідомий тип моделі: {model_type}")
            
        return y_trend, coeffs, model_type
    
    def generate_synthetic_data(self, y_trend: np.ndarray, residuals: np.ndarray, distribution: str = 'normal') -> np.ndarray:
        """
        Генерація синтетичних даних на основі тренду та розподілу залишків
        
        Args:
            y_trend: Тренд моделі
            residuals: Залишки (y_actual - y_trend) для обчислення стандартного відхилення
            distribution: Тип розподілу шуму ('normal' | 'uniform')
            
        Returns:
            Синтетичні дані (y_trend + noise)
        """
        std = np.std(residuals, ddof=1)
        
        if distribution == 'normal':
            noise = np.random.normal(loc=0, scale=std, size=len(y_trend))
        elif distribution == 'uniform':
            delta = std * np.sqrt(3)
            noise = np.random.uniform(low=-delta, high=delta, size=len(y_trend))
        else:
            raise ValueError(f"Невідомий тип розподілу: {distribution}")
        
        return y_trend + noise
    
    def find_optimal_polynomial_degree(self) -> Tuple[int, Dict]:
        """
        Етап 3: Визначення оптимального порядку полінома
        за критерієм мінімуму різниці СКВ похідних (термін №35)
        
        Returns:
            (optimal_degree, diagnostics)
        """
        results = {}
        
        for m in range(1, self.max_degree + 1):
            if self.n < 2 * m:
                break
            
            coeffs = np.polyfit(self.X, self.y, m)
            poly = np.poly1d(coeffs)
            y_fit = poly(self.X)
            
            residuals = self.y - y_fit
            
            sigma_exp = np.std(residuals, ddof=m+1)
            
            p = m
            poly_deriv = None
            if p > 0:
                poly_deriv = poly.deriv(p)
                
                from math import factorial
                sigma_theor = sigma_exp * factorial(p)
            else:
                sigma_theor = sigma_exp
            
            if p > 0 and p < len(self.y) - 1 and poly_deriv is not None:
                y_deriv_exp = self._numerical_derivative(self.y, p)
                y_deriv_theor = poly_deriv(self.X[p:])
                
                if len(y_deriv_exp) == len(y_deriv_theor):
                    sigma_deriv_exp = np.std(y_deriv_exp - y_deriv_theor)
                else:
                    sigma_deriv_exp = sigma_exp
            else:
                sigma_deriv_exp = sigma_exp
            
            delta = np.abs(sigma_deriv_exp - sigma_theor)
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.y - np.mean(self.y))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            adj_r2 = 1 - ((1 - r2) * (self.n - 1) / (self.n - m - 1))
            
            results[m] = {
                'degree': m,
                'delta': delta,
                'r2': r2,
                'adj_r2': adj_r2,
                'sigma_exp': sigma_exp,
                'sigma_theor': sigma_theor,
                'coeffs': coeffs
            }
        
        if not results:
            return 1, {}
        
        valid = {k: v for k, v in results.items() if v['adj_r2'] > 0.7}
        
        if not valid:
            optimal_m = max(results.keys(), key=lambda k: results[k]['r2'])
        else:
            optimal_m = min(valid.keys(), key=lambda k: valid[k]['delta'])
        
        return optimal_m, results
    
    def _numerical_derivative(self, y: np.ndarray, order: int) -> np.ndarray:
        """Числове диференціювання порядку order"""
        result = y.copy()
        for _ in range(order):
            result = np.diff(result)
        return result
    
    def synthesize(self) -> Dict:
        """
        Повний цикл синтезу математичної моделі
        
        Returns:
            dict: {
                'model_type': str ('poly' | 'log' | 'exp'),
                'degree': int (для poly),
                'coeffs': np.ndarray,
                'trend_info': dict,
                'diagnostics': dict,
                'recommended_distribution': str
            }
        """
        print("="*60)
        print("ЕТАП 4.2: СИНТЕЗ МАТЕМАТИЧНОЇ МОДЕЛІ")
        print("="*60)
        
        print("\n[1/3] Аналіз характеру тренду...")
        trend_info = self.analyze_trend_type()
        print(f"    Тип тренду: {trend_info['type']}")
        print(f"    Монотонність: {trend_info['monotonic_score']:.2f}")
        print(f"    Сезонність: {'Так' if trend_info['seasonality_detected'] else 'Ні'}")
        
        print("\n[2/3] Вибір класу математичної моделі...")
        model_type = self.select_model_class(trend_info)
        print(f"    Обрано: {model_type}")
        
        print("\n[3/3] Визначення оптимальних параметрів...")
        
        degree = None
        coeffs = np.array([])
        diagnostics = {}
        
        if model_type == 'poly':
            degree, diagnostics = self.find_optimal_polynomial_degree()
            coeffs = diagnostics[degree]['coeffs']
            print(f"    Оптимальний порядок полінома: {degree}")
            print(f"    R²: {diagnostics[degree]['r2']:.4f}")
            print(f"    Adj R²: {diagnostics[degree]['adj_r2']:.4f}")
            
            print("\n    Порівняння порядків полінома:")
            print(f"    {'m':<4} | {'R²':<8} | {'Adj R²':<8} | {'Δ(σ)':<10}")
            print("    " + "-"*40)
            for m in sorted(diagnostics.keys()):
                d = diagnostics[m]
                mark = " ← *" if m == degree else ""
                print(f"    {m:<4} | {d['r2']:<8.4f} | {d['adj_r2']:<8.4f} | {d['delta']:<10.4f}{mark}")
        
        elif model_type == 'log':
            X_log = np.log(self.X[1:] + 1)
            y_subset = self.y[1:]
            coeffs = np.polyfit(X_log, y_subset, 1)
            diagnostics = {'type': 'log'}
            print(f"    Коефіцієнти: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}")
            print(f"    Модель: y = {coeffs[0]:.4f}·ln(t) + {coeffs[1]:.4f}")
        
        elif model_type == 'exp':
            y_log = np.log(self.y)
            coeffs_log = np.polyfit(self.X, y_log, 1)
            coeffs = np.array([np.exp(coeffs_log[1]), coeffs_log[0]])
            diagnostics = {'type': 'exp'}
            print(f"    Коефіцієнти: A={coeffs[0]:.4f}, k={coeffs[1]:.4f}")
            print(f"    Модель: y = {coeffs[0]:.4f}·exp({coeffs[1]:.4f}·t)")
        
        print("\n" + "="*60)
        print("СИНТЕЗ ЗАВЕРШЕНО")
        print("="*60 + "\n")
        
        # Автоматичне визначення типу розподілу шуму
        y_trend, _, _ = self.build_trend(model_type, degree, coeffs)
        residuals = self.y - y_trend
        
        from scipy import stats
        _, p_value = stats.shapiro(residuals)
        
        # Якщо p-value > 0.05, розподіл близький до нормального
        recommended_distribution = 'normal' if p_value > 0.05 else 'uniform'
        
        return {
            'model_type': model_type,
            'degree': degree,
            'coeffs': coeffs,
            'trend_info': trend_info,
            'diagnostics': diagnostics,
            'recommended_distribution': recommended_distribution
        }