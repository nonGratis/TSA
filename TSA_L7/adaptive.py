import numpy as np
from collections import deque

class NISAdapter:
    """
    Адаптація Q на основі NIS.
    Стабільна версія: плавна зміна, швидке відновлення.
    """
    def __init__(
        self,
        dof: int = 1,
        window_size: int = 5,
        scale_factor: float = 1.5,
        decay_factor: float = 0.8 
    ):
        self.window = deque(maxlen=window_size)
        self.high_threshold = 4.0 # Chi2 ~95%
        self.scale_factor = scale_factor
        self.decay_factor = decay_factor
        self.multiplier = 1.0

    def update(self, residual: float, S: float, current_q: float, base_q: float) -> float:
        if S <= 1e-9: return current_q
            
        nis = (residual ** 2) / S
        self.window.append(nis)
        
        mean_nis = np.mean(self.window)
        
        if mean_nis > self.high_threshold:
            # Інфляція
            ratio = mean_nis / self.high_threshold
            step_up = 1.0 + (ratio - 1.0) * 0.1 # Плавний крок
            self.multiplier = min(self.multiplier * step_up, 20.0)
        else:
            # Дефляція
            self.multiplier = max(1.0, self.multiplier * self.decay_factor)
            
        return float(base_q * self.multiplier)

    def get_statistics(self) -> dict:
        return {
            'mean_nis': np.mean(self.window) if self.window else 0.0,
            'multiplier': self.multiplier
        }