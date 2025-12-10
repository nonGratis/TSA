import numpy as np
from collections import deque
from typing import Optional

class AdaptiveAlpha:
    """
    Адаптивне налаштування параметру Alpha (smoothing factor).
    
    Логіка:
    - Якщо помилка прогнозу (residual) зростає -> збільшуємо Alpha (довіряємо вимірам, менше згладжування).
    - Якщо помилка мала і стабільна -> зменшуємо Alpha (більше згладжування, краще пригнічення шуму).
    """
    def __init__(
        self,
        window: int = 12,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        base_alpha: float = 0.5
    ):
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.current_alpha = base_alpha
        
        self.residuals = deque(maxlen=window)
        
        # Статистика
        self.increase_count = 0
        self.decrease_count = 0
        
    def update(self, residual: float, expected_noise_r: float) -> float:
        """
        Повертає нове значення Alpha.
        
        Args:
            residual: поточна помилка передбачення.
            expected_noise_r: очікувана дисперсія шуму вимірювання (R).
        """
        self.residuals.append(residual)
        
        if len(self.residuals) < 3:
            return self.current_alpha
            
        # Поточна середня квадратична помилка (MSE) у вікні
        current_mse = np.mean(np.square(self.residuals))
        
        # Відношення поточної помилки до очікуваного шуму (Signal-to-Noise estimate)
        # Якщо MSE >> R, значить відбувається маневр -> треба швидка реакція (велике Alpha)
        # Якщо MSE <= R, значить ми в межах шуму -> треба фільтрація (мале Alpha)
        
        ratio = current_mse / (expected_noise_r + 1e-9)
        
        # Емпірична функція відображення Ratio -> Alpha
        # Sigmoid-like mapping or simple linear clamp
        # Target alpha based on ratio
        if ratio > 9.0: # 3-sigma deviation
            target = self.alpha_max
        elif ratio < 1.0:
            target = self.alpha_min
        else:
            # Log scale mapping between 1 and 9
            norm = (np.log(ratio) - np.log(1.0)) / (np.log(9.0) - np.log(1.0)) # 0..1
            target = self.alpha_min + norm * (self.alpha_max - self.alpha_min)
            
        # Плавне оновлення (Exponential Moving Average для самого Alpha)
        # Щоб Alpha не стрибала як скажена
        smoothing = 0.2
        new_alpha = self.current_alpha * (1 - smoothing) + target * smoothing
        
        # Статистика
        if new_alpha > self.current_alpha + 0.01:
            self.increase_count += 1
        elif new_alpha < self.current_alpha - 0.01:
            self.decrease_count += 1
            
        self.current_alpha = new_alpha
        return self.current_alpha

    def get_statistics(self) -> dict:
        return {
            'alpha_current': self.current_alpha,
            'increase_count': self.increase_count,
            'decrease_count': self.decrease_count
        }