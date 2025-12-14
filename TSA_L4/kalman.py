import numpy as np
from typing import Optional, Tuple

def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """Оцінка Q (Process) та R (Measurement) через дисперсії різниць."""
    y_clean = y[np.isfinite(y)]
    if len(y_clean) < 10: return 1.0, 10.0

    diffs = np.diff(y_clean)
    measurement_noise = np.var(diffs) if len(diffs) > 0 else 1.0

    if len(y_clean) >= 5:
        second_diffs = np.diff(diffs)
        process_noise = np.var(second_diffs) if len(second_diffs) > 0 else 0.1
    else:
        process_noise = measurement_noise * 0.1

    return float(max(process_noise, 1e-4)), float(max(measurement_noise, 1e-2))


class AlphaBetaFilter:
    """
    Класичний скалярний Alpha-Beta(-Gamma) фільтр.
    Реалізація за формулами Benedict-Bordner з тюнінгом чутливості.
    """
    def __init__(
        self,
        dt: float = 1.0,
        state_dim: int = 2,
        process_noise_q: float = 1.0,
        measurement_noise_r: float = 1.0,
        init_state: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ):
        self.dt = float(dt)
        self.state_dim = state_dim
        self.Q = float(process_noise_q)
        self.R = float(measurement_noise_r)

        self.x = 0.0
        self.v = 0.0
        self.a = 0.0

        if init_state is not None:
            safe = np.nan_to_num(init_state)
            self.x = float(safe[0])
            if len(safe) > 1: self.v = float(safe[1])
            if len(safe) > 2 and state_dim == 3: self.a = float(safe[2])

        if alpha is not None:
            self.alpha = float(alpha)
            self._recalc_gains_from_alpha()
        else:
            self.update_params_from_noise(self.Q, self.R)

    def update_params_from_noise(self, q: float, r: float) -> None:
        """
        Розрахунок оптимальних Alpha/Beta.
        Використовується агресивний Tracking Index для зменшення лагу.
        """
        self.Q = max(float(q), 1e-9)
        self.R = max(float(r), 1e-9)
        
        # TUNING: Множник 2.0 для збільшення чутливості (зменшення Bias)
        lam = (self.Q / self.R) * (self.dt ** 2) * 2.0

        if self.state_dim == 2:
            # Benedict-Bordner
            r_val = (4.0 + lam - np.sqrt(8.0 * lam + lam**2)) / 4.0
            self.alpha = 1.0 - r_val**2
            self.beta = 2.0 * (2.0 - self.alpha) - 4.0 * np.sqrt(1.0 - self.alpha)
            
            # CRITICAL: Мінімальна Alpha = 0.2 для уникнення інерційності
            self.alpha = np.clip(self.alpha, 0.2, 0.99)
            self.gamma = 0.0
        else:
            # Емпіричне наближення для CA
            self.alpha = np.clip(0.6 * np.sqrt(lam), 0.2, 0.95)
            self.beta = 2.0 * self.alpha
            self.gamma = 0.5 * (self.alpha ** 2)

    def _recalc_gains_from_alpha(self):
        if self.state_dim == 2:
            self.beta = 2 * (2 - np.sqrt(1 - self.alpha))
            self.gamma = 0.0
        else:
            self.beta = 2.0 * self.alpha
            self.gamma = 0.5 * (self.alpha ** 2)

    def set_alpha(self, alpha: float):
        self.alpha = np.clip(alpha, 0.001, 0.99)
        self._recalc_gains_from_alpha()

    def predict(self) -> float:
        if self.state_dim == 2:
            self.x = self.x + self.v * self.dt
        else:
            self.x = self.x + self.v * self.dt + 0.5 * self.a * (self.dt**2)
            self.v = self.v + self.a * self.dt
        return self.x

    def update(self, measurement: float) -> float:
        residual = measurement - self.x
        
        self.x = self.x + self.alpha * residual
        self.v = self.v + (self.beta / self.dt) * residual
        
        if self.state_dim == 3:
            self.a = self.a + (self.gamma / (self.dt**2)) * residual
            
        return self.x

    def get_residual(self, measurement: float) -> float:
        return measurement - self.x

    def get_innovation_variance(self) -> float:
        return self.R / (1.0 - self.alpha + 1e-6)

    # Гетери
    def get_position(self): return self.x
    def get_velocity(self): return self.v
    def get_acceleration(self): return self.a
    def get_position_variance(self): return self.R * self.alpha 

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros(k)
        vars_pred = np.zeros(k)
        
        curr_x, curr_v, curr_a = self.x, self.v, self.a
        
        for i in range(k):
            t = (i + 1) * self.dt
            if self.state_dim == 2:
                preds[i] = curr_x + curr_v * t
            else:
                preds[i] = curr_x + curr_v * t + 0.5 * curr_a * (t**2)
            
            vars_pred[i] = self.R * (1 + i*0.2) 
            
        return preds, vars_pred