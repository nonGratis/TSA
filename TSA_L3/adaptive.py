import numpy as np
from collections import deque
from scipy.stats import chi2

class NISAdapter:
    """
    Адаптація шуму процесу (Q) на основі статистики NIS (Normalized Innovation Squared).
    
    Академічне обґрунтування:
    NIS = ε_k^T * S_k^(-1) * ε_k, де ε - інновація, S - коваріація інновації.
    При коректній роботі фільтра NIS підпорядковується розподілу Хі-квадрат (χ²).
    Якщо E[NIS] >> m (розмірність виміру), це свідчить про розходження фільтра
    або маневр цілі.
    
    Стратегія: Інфляція Q при детекції розходження.
    """
    def __init__(
        self,
        dof: int = 1,
        alpha_significance: float = 0.05,
        window_size: int = 12,
        scale_factor: float = 5.0,
        decay_factor: float = 0.95
    ):
        """
        Args:
            dof: Ступені свободи (розмірність виміру).
            alpha_significance: Рівень значущості (0.05 -> 95% confidence).
            window_size: Вікно усереднення NIS для робастності.
        """
        self.window = deque(maxlen=window_size)
        
        # Критичне значення χ² для перевірки гіпотези
        self.chi2_threshold_high = chi2.ppf(1 - alpha_significance, df=dof)
        # Нижня межа (опціонально для зменшення Q, тут не використовується агресивно)
        self.chi2_threshold_low = chi2.ppf(alpha_significance, df=dof)
        
        self.scale_factor = scale_factor
        self.decay_factor = decay_factor
        self.base_q_multiplier = 1.0

    def update(self, residual: float, S: float, current_q: float, base_q: float) -> float:
        """
        Розрахунок нового Q на основі NIS.
        
        Args:
            residual: Інновація (y - Hx).
            S: Теоретична коваріація інновації (HPHT + R).
            current_q: Поточне значення Q.
            base_q: Базове значення Q (дизайн-параметр).
            
        Returns:
            Нове значення Q.
        """
        if S <= 1e-9:
            return float(current_q)
            
        # 1. Розрахунок NIS
        nis = (residual ** 2) / S
        self.window.append(nis)
        
        # 2. Усереднення (Sample Mean NIS)
        mean_nis = np.mean(self.window)
        
        # 3. Перевірка гіпотези (Divergence Test)
        if mean_nis > self.chi2_threshold_high:
            # Фільтр розходиться (недооцінює динаміку) -> Inflation Q
            # Масштабуємо Q пропорційно відхиленню NIS
            scaling = (mean_nis / self.chi2_threshold_high) * self.scale_factor
            self.base_q_multiplier = max(self.base_q_multiplier, scaling)
        else:
            # Фільтр узгоджений -> Decay до базового рівня
            self.base_q_multiplier = max(1.0, self.base_q_multiplier * self.decay_factor)
            
        # FIX: Явне приведення до float для задоволення вимог типізації
        return float(base_q * self.base_q_multiplier)

    def get_statistics(self) -> dict:
        return {
            'mean_nis': np.mean(self.window) if self.window else 0.0,
            'q_multiplier': self.base_q_multiplier,
            'chi2_limit': self.chi2_threshold_high
        }