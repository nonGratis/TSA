import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class SyntheticTimeSeriesGenerator:
    """
    Генератор синтетичних часових рядів з заданими властивостями.
    """
    
    def __init__(self, length: int = 1000, random_state: Optional[int] = None):
        """
        Args:
            length: Довжина часового ряду
            random_state: Seed для відтворюваності
        """
        self.length = length
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate_trend(self, trend_type: str = 'linear', 
                      params: Optional[Dict] = None) -> np.ndarray:
        """
        Генерація трендової компоненти.
        
        Args:
            trend_type: Тип тренду ('linear', 'polynomial', 'exponential', 'logarithmic')
            params: Параметри тренду
            
        Returns:
            Масив значень тренду
        """
        if params is None:
            params = {}
        
        t = np.arange(self.length)
        
        if trend_type == 'linear':
            slope = params.get('slope', 0.1)
            intercept = params.get('intercept', 0.0)
            trend = slope * t + intercept
            
        elif trend_type == 'polynomial':
            coeffs = params.get('coeffs', [0.0001, 0.01, 0])
            trend = np.polyval(coeffs, t)
            
        elif trend_type == 'exponential':
            rate = params.get('rate', 0.001)
            base = params.get('base', 1.0)
            trend = base * np.exp(rate * t)
            
        elif trend_type == 'logarithmic':
            scale = params.get('scale', 10.0)
            trend = scale * np.log(t + 1)
            
        else:
            raise ValueError(f"Невідомий тип тренду: {trend_type}")
        
        return trend
    
    def generate_seasonality(self, periods: list = [24], 
                           amplitudes: Optional[list] = None) -> np.ndarray:
        """
        Генерація сезонної компоненти (суперпозиція синусоїд).
        
        Args:
            periods: Список періодів сезонності
            amplitudes: Список амплітуд для кожного періоду
            
        Returns:
            Масив значень сезонності
        """
        if amplitudes is None:
            amplitudes = [1.0] * len(periods)
        
        if len(amplitudes) != len(periods):
            raise ValueError("Кількість амплітуд має дорівнювати кількості періодів")
        
        t = np.arange(self.length)
        seasonal = np.zeros(self.length)
        
        for period, amplitude in zip(periods, amplitudes):
            phase = np.random.uniform(0, 2*np.pi)
            seasonal += amplitude * np.sin(2 * np.pi * t / period + phase)
        
        return seasonal
    
    def generate_noise(self, noise_type: str = 'white', 
                      std: float = 1.0, 
                      params: Optional[Dict] = None) -> np.ndarray:
        """
        Генерація шумової компоненти.
        
        Args:
            noise_type: Тип шуму ('white', 'pink', 'brown', 'ar')
            std: Стандартне відхилення
            params: Додаткові параметри
            
        Returns:
            Масив значень шуму
        """
        if params is None:
            params = {}
        
        if noise_type == 'white':
            # Білий шум (незалежні нормальні величини)
            noise = np.random.normal(0, std, self.length)
            
        elif noise_type == 'pink':
            # Рожевий шум (1/f шум)
            noise = self._generate_pink_noise(std)
            
        elif noise_type == 'brown':
            # Коричневий шум (інтеграл білого шуму)
            white = np.random.normal(0, std, self.length)
            noise = np.cumsum(white)
            noise = noise / np.std(noise) * std  # Нормалізація
            
        elif noise_type == 'ar':
            # AR(1) процес
            phi = params.get('phi', 0.7)  # Коефіцієнт автокореляції
            noise = self._generate_ar1(phi, std)
            
        else:
            raise ValueError(f"Невідомий тип шуму: {noise_type}")
        
        return noise
    
    def _generate_pink_noise(self, std: float) -> np.ndarray:
        """Генерація рожевого (1/f) шуму через FFT."""
        # Генеруємо білий шум
        white = np.random.randn(self.length)
        
        # FFT
        fft_white = np.fft.rfft(white)
        
        # Створюємо 1/f фільтр
        freqs = np.fft.rfftfreq(self.length)
        freqs[0] = 1  # Уникаємо ділення на 0
        filt = 1 / np.sqrt(freqs)
        
        # Застосовуємо фільтр
        fft_pink = fft_white * filt
        
        # Inverse FFT
        pink = np.fft.irfft(fft_pink, self.length)
        
        # Нормалізація
        pink = pink / np.std(pink) * std
        
        return pink
    
    def _generate_ar1(self, phi: float, std: float) -> np.ndarray:
        """Генерація AR(1) процесу."""
        noise = np.zeros(self.length)
        noise[0] = np.random.normal(0, std)
        
        for t in range(1, self.length):
            noise[t] = phi * noise[t-1] + np.random.normal(0, std * np.sqrt(1 - phi**2))
        
        return noise
    
    def generate_with_hurst(self, hurst: float, std: float = 1.0) -> np.ndarray:
        """
        Генерація часового ряду з заданим показником Hurst.
        
        Args:
            hurst: Показник Hurst (0 < H < 1)
            std: Стандартне відхилення
            
        Returns:
            Fractional Brownian Motion
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst має бути між 0 і 1")
        
        # fBm через спектральний метод
        # Генеруємо частоти
        freqs = np.fft.rfftfreq(self.length)[1:]  # Пропускаємо 0
        
        # Спектральна щільність: S(f) ~ f^(-2H-1)
        spectrum = freqs ** (-(2*hurst + 1))
        
        # Генеруємо випадкові фази
        phases = np.random.uniform(0, 2*np.pi, len(freqs))
        
        # Створюємо комплексні амплітуди
        amplitudes = np.sqrt(spectrum) * np.exp(1j * phases)
        amplitudes = np.concatenate([[0], amplitudes])  # Додаємо DC компоненту
        
        # Inverse FFT
        fbm = np.fft.irfft(amplitudes, self.length)
        
        # Нормалізація
        fbm = fbm / np.std(fbm) * std
        
        return fbm
    
    def generate_combined(self, 
                         trend_params: Optional[Dict] = None,
                         seasonal_params: Optional[Dict] = None,
                         noise_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Генерація комбінованого часового ряду (тренд + сезонність + шум).
        
        Args:
            trend_params: {'type': 'linear', 'slope': 0.1, ...}
            seasonal_params: {'periods': [24], 'amplitudes': [1.0]}
            noise_params: {'type': 'white', 'std': 1.0}
            
        Returns:
            (combined_series, components_dict)
        """
        if trend_params is None:
            trend_params = {'type': 'linear', 'slope': 0.1}
        if seasonal_params is None:
            seasonal_params = {'periods': [24], 'amplitudes': [5.0]}
        if noise_params is None:
            noise_params = {'type': 'white', 'std': 1.0}
        
        # Генеруємо компоненти
        trend = self.generate_trend(
            trend_params.get('type', 'linear'),
            trend_params
        )
        
        seasonal = self.generate_seasonality(
            seasonal_params.get('periods', [24]),
            seasonal_params.get('amplitudes', [1.0])
        )
        
        noise = self.generate_noise(
            noise_params.get('type', 'white'),
            noise_params.get('std', 1.0),
            noise_params
        )
        
        # Комбінуємо
        combined = trend + seasonal + noise
        
        components = {
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'combined': combined
        }
        
        return combined, components
    
    def generate_from_real_properties(self, real_data: pd.Series, 
                                     decomposition_result: Dict,
                                     properties: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Генерація синтетичних даних на основі властивостей реальних даних.
        
        Args:
            real_data: Реальний часовий ряд
            decomposition_result: Результат декомпозиції
            properties: Результат аналізу властивостей
            
        Returns:
            (synthetic_series, info_dict)
        """
        print("\n=== ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ ===\n")
        
        # Витягуємо властивості
        trend_component = decomposition_result['trend'].values
        seasonal_component = decomposition_result['seasonal'].values
        
        # Параметри тренду (апроксимація поліномом 2-го степеня)
        t = np.arange(len(trend_component))
        valid_mask = np.isfinite(trend_component)
        trend_coeffs = np.polyfit(t[valid_mask], trend_component[valid_mask], 2)
        
        print(f"  Тренд: поліном 2-го степеня")
        print(f"    Коефіцієнти: {trend_coeffs}")
        
        # Параметри сезонності (FFT для виділення головних частот)
        from scipy.fft import fft, fftfreq
        
        seasonal_clean = seasonal_component[np.isfinite(seasonal_component)]
        fft_vals = fft(seasonal_clean)
        freqs = fftfreq(len(seasonal_clean))
        
        # Знаходимо домінуючі частоти
        power = np.abs(fft_vals)
        idx = np.argsort(power)[::-1][1:4]  # Топ-3 (пропускаємо DC)
        
        periods = [int(1 / abs(freqs[i])) for i in idx if freqs[i] != 0]
        amplitudes = [power[i] / len(seasonal_clean) * 2 for i in idx]
        
        print(f"  Сезонність: {len(periods)} домінуючих періодів")
        print(f"    Періоди: {periods[:3]}")
        print(f"    Амплітуди: {[f'{a:.2f}' for a in amplitudes[:3]]}")
        
        # Параметри шуму
        hurst_val = properties['hurst'].get('hurst', 0.5)
        noise_std = np.std(decomposition_result['resid'].dropna())
        
        print(f"  Шум: Hurst = {hurst_val:.3f}, σ = {noise_std:.3f}")
        
        # Генеруємо синтетичні дані
        synthetic_trend = np.polyval(trend_coeffs, t)
        
        synthetic_seasonal = self.generate_seasonality(
            periods[:3], 
            amplitudes[:3]
        )
        
        if hurst_val is not None and 0.1 < hurst_val < 0.9:
            synthetic_noise = self.generate_with_hurst(hurst_val, noise_std)
        else:
            synthetic_noise = self.generate_noise('white', noise_std)
        
        synthetic_combined = synthetic_trend + synthetic_seasonal + synthetic_noise
        
        print(f"\n  Згенеровано {self.length} точок")
        print("="*50 + "\n")
        
        return synthetic_combined, {
            'trend': synthetic_trend,
            'seasonal': synthetic_seasonal,
            'noise': synthetic_noise,
            'combined': synthetic_combined,
            'params': {
                'trend_coeffs': trend_coeffs,
                'periods': periods[:3],
                'amplitudes': amplitudes[:3],
                'hurst': hurst_val,
                'noise_std': noise_std
            }
        }


def generate_test_series(n: int = 1000, 
                        trend: str = 'linear',
                        seasonal_periods: list = [24],
                        noise_type: str = 'white',
                        random_state: int = 42) -> pd.Series:
    """
    Швидка генерація тестового часового ряду.
    """
    generator = SyntheticTimeSeriesGenerator(n, random_state)
    
    trend_params = {'type': trend, 'slope': 0.05}
    seasonal_params = {'periods': seasonal_periods, 'amplitudes': [5.0] * len(seasonal_periods)}
    noise_params = {'type': noise_type, 'std': 1.0}
    
    combined, _ = generator.generate_combined(trend_params, seasonal_params, noise_params)
    
    # Створюємо pandas Series з datetime індексом
    dates = pd.date_range(start='2024-01-01', periods=n, freq='h')
    series = pd.Series(combined, index=dates)
    
    return series