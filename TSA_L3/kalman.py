import numpy as np
from typing import Optional, Tuple


def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """Оцінка Q та R методом дисперсій різниць."""
    if len(y) < 2:
        return 1.0, 1.0

    diffs = np.diff(y)
    measurement_noise = float(np.var(diffs, ddof=1)) # R

    if len(y) >= 3:
        second_diffs = np.diff(diffs)
        process_noise = float(np.var(second_diffs, ddof=1)) # Q
    else:
        process_noise = measurement_noise / 10.0

    return max(process_noise, 1e-6), max(measurement_noise, 1e-6)


class AlphaBetaFilter:
    """
    Steady-State Kalman Filter (Alpha-Beta(-Gamma)).
    Використовує аналітичний розв'язок рівняння Ріккаті для стаціонарного режиму.
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
        if state_dim not in [2, 3]:
            raise ValueError("state_dim must be 2 or 3")
        
        self.dt = dt
        self.state_dim = state_dim
        
        # State Vector [x, v, a]
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0
        
        if init_state is not None:
            self.x = float(init_state[0])
            if len(init_state) > 1: self.v = float(init_state[1])
            if len(init_state) > 2 and state_dim == 3: self.a = float(init_state[2])

        self.R = measurement_noise_r
        self.Q = process_noise_q
        
        # Gains
        self.alpha = 0.1
        self.beta = 0.01
        self.gamma = 0.0
        
        if alpha is not None:
            self.set_alpha(alpha)
        else:
            self.update_params_from_noise(process_noise_q, measurement_noise_r)

    def update_params_from_noise(self, q: float, r: float) -> None:
        """
        Розрахунок оптимальних Alpha/Beta через Tracking Index (Lambda).
        Lambda = (Q * dt^k) / R.
        """
        self.Q = q
        self.R = max(r, 1e-9)
        
        if self.state_dim == 2:
            # Lambda for Constant Velocity
            lambda_idx = np.sqrt(self.Q / self.R) * (self.dt ** 2)
            
            # Optimal solution (Kalman stationary gain)
            r_param = (4 + lambda_idx - np.sqrt(8 * lambda_idx + lambda_idx**2)) / 4
            self.alpha = 1 - r_param**2
            self.beta = 2 * (2 - self.alpha) - 4 * np.sqrt(1 - self.alpha)
            self.gamma = 0.0
            
        else: # dim == 3
            # Lambda for Constant Acceleration (Jezek approx)
            lambda_idx = (self.Q / self.R) * (self.dt ** 4)
            b = lambda_idx / 2.0
            self.alpha = 1 - (1.0 / (1 + b))**3
            
            # Stability constraints
            self.alpha = np.clip(self.alpha, 0.001, 0.999)
            self.beta = 2 * (2 - np.sqrt(1 - self.alpha))
            self.gamma = (self.beta**2) / (2 * self.alpha)

    def set_alpha(self, new_alpha: float) -> None:
        """Ручне встановлення alpha зі збереженням зв'язків стабільності."""
        self.alpha = np.clip(new_alpha, 0.001, 0.999)
        if self.state_dim == 2:
            self.beta = 2 * (2 - self.alpha) - 4 * np.sqrt(1 - self.alpha)
        else:
            self.beta = 2 * (2 - np.sqrt(1 - self.alpha))
            self.gamma = (self.beta**2) / (2 * self.alpha)

    def predict(self) -> float:
        if self.state_dim == 2:
            self.x += self.v * self.dt
        else:
            self.x += self.v * self.dt + 0.5 * self.a * self.dt**2
            self.v += self.a * self.dt
        return self.x

    def update(self, measurement: float) -> float:
        residual = measurement - self.x
        
        self.x += self.alpha * residual
        self.v += (self.beta / self.dt) * residual
        
        if self.state_dim == 3:
            self.a += (self.gamma / (0.5 * self.dt**2)) * residual
            
        return self.x

    def get_residual(self, measurement: float) -> float:
        return measurement - self.x

    def get_position(self) -> float: return self.x
    def get_velocity(self) -> float: return self.v
    def get_acceleration(self) -> float: return self.a

    def get_position_variance(self) -> float:
        """Теоретична апостеріорна дисперсія позиції (P_ss)."""
        if self.state_dim == 2:
            denom = self.alpha * (4 - 2*self.alpha - self.beta)
            if abs(denom) < 1e-9: denom = 1e-9
            num = (2 * self.alpha**2 + 2 * self.beta + self.alpha * self.beta)
            k_gain = num / denom
        else:
            k_gain = self.alpha / (1 - self.alpha) # Approx
        return k_gain * self.R
    
    def get_innovation_variance(self) -> float:
        """
        Теоретична коваріація інновації S.
        S = H * P_pred * H^T + R
        Для Alpha-Beta (скаляр): S = P_pred + R
        P_pred ≈ P_post / (1 - alpha) (грубе наближення для стаціонарного режиму)
        Точніше: S = R / (1 - K*H) = R / (1 - alpha)
        """
        return self.R / (1.0 - self.alpha + 1e-9)

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros(k)
        vars_pred = np.zeros(k)
        
        curr_x, curr_v, curr_a = self.x, self.v, self.a
        p0 = self.get_position_variance()
        
        for i in range(k):
            t = (i + 1) * self.dt
            if self.state_dim == 2:
                preds[i] = curr_x + curr_v * t
                vars_pred[i] = p0 + (t**2) * (p0 * 0.1)
            else:
                preds[i] = curr_x + curr_v * t + 0.5 * curr_a * t**2
                vars_pred[i] = p0 + (t**4) * (p0 * 0.01)
                
        return preds, vars_pred