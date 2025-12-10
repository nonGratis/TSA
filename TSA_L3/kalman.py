import numpy as np
from typing import Optional, Tuple


def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """
    Оцінка параметрів шуму (залишаємо для сумісності та розрахунку індексу λ)
    """
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
    Скалярний фільтр Alpha-Beta (для state_dim=2) або Alpha-Beta-Gamma (для state_dim=3).
    
    Це спрощена версія фільтра Калмана для стаціонарних умов.
    Він не використовує матричні операції.
    """
    def __init__(
        self,
        dt: float = 1.0,
        state_dim: int = 2,
        process_noise_q: float = 1.0,
        measurement_noise_r: float = 1.0,
        init_state: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        if state_dim not in [2, 3]:
            raise ValueError("state_dim має бути 2 або 3")
        
        self.dt = dt
        self.state_dim = state_dim
        
        # Ініціалізація стану [x, v, a]
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0 # Використовується тільки при dim=3
        
        if init_state is not None:
            self.x = float(init_state[0])
            if len(init_state) > 1:
                self.v = float(init_state[1])
            if len(init_state) > 2 and state_dim == 3:
                self.a = float(init_state[2])

        # Зберігаємо R для розрахунку теоретичної дисперсії помилки
        self.R = measurement_noise_r
        
        # Розрахунок оптимальних alpha/beta/gamma, якщо не задані вручну
        # Використовуємо Tracking Index Lambda = (Q * dt^k) / R
        if alpha is None:
            self.update_params_from_noise(process_noise_q, measurement_noise_r)
        else:
            self.alpha = alpha
            self.beta = beta if beta is not None else 0.1
            self.gamma = gamma if gamma is not None else 0.01

    def update_params_from_noise(self, q: float, r: float) -> None:
        """
        Розрахунок оптимальних параметрів фільтра на основі співвідношення шумів.
        Використовуються аналітичні рішення для стаціонарного фільтра Калмана.
        """
        self.R = r
        if r < 1e-9: r = 1e-9
        
        # Tracking Index
        lambda_idx = np.sqrt(q / r) * (self.dt ** 2)
        
        if self.state_dim == 2:
            # Optimal Alpha-Beta relationships
            # r_param - допоміжна змінна
            r_param = (4 + lambda_idx - np.sqrt(8 * lambda_idx + lambda_idx**2)) / 4
            self.alpha = 1 - r_param**2
            self.beta = 2 * (2 - self.alpha) - 4 * np.sqrt(1 - self.alpha)
            self.gamma = 0.0
            
        else: # state_dim == 3 (Alpha-Beta-Gamma)
            # Для ABG lambda визначається інакше (через Q прискорення)
            lambda_idx = (q / r) * (self.dt ** 4) # approx scaling
            # Емпіричне наближення для ABG (Jezek's approximation or similar)
            b = lambda_idx / 2.0
            self.alpha = 1 - (1.0 / (1 + b))**3 # Crude approx, better to use iterative but this fits "simple filter"
            
            # Ensure stability constraints
            self.alpha = np.clip(self.alpha, 0.01, 0.99)
            self.beta = 2 * (2 - np.sqrt(1-self.alpha)) # Standard constraint relationship
            self.gamma = (self.beta**2) / (2*self.alpha)

    def set_alpha(self, new_alpha: float) -> None:
        """
        Ручне встановлення alpha (для адаптивного режиму).
        Beta і Gamma перераховуються для збереження стабільності.
        """
        self.alpha = np.clip(new_alpha, 0.001, 0.999)
        
        if self.state_dim == 2:
            # Зв'язок для критично демпфованого фільтра
            self.beta = 2 * (2 - self.alpha) - 4 * np.sqrt(1 - self.alpha)
        else:
            # Зв'язок для ABG
            self.beta = 2 * (2 - np.sqrt(1 - self.alpha)) # спрощено
            self.gamma = (self.beta**2) / (2 * self.alpha)

    def predict(self) -> float:
        """Екстраполяція стану"""
        if self.state_dim == 2:
            self.x = self.x + self.v * self.dt
            # self.v не змінюється (constant velocity model)
        else:
            self.x = self.x + self.v * self.dt + 0.5 * self.a * self.dt**2
            self.v = self.v + self.a * self.dt
            # self.a не змінюється (constant acceleration model)
        return self.x

    def update(self, measurement: float) -> float:
        """Корекція стану за виміром"""
        residual = measurement - self.x
        
        self.x = self.x + self.alpha * residual
        self.v = self.v + (self.beta / self.dt) * residual
        
        if self.state_dim == 3:
            self.a = self.a + (self.gamma / (0.5 * self.dt**2)) * residual
            
        return self.x

    def step(self, measurement: float) -> float:
        self.predict()
        return self.update(measurement)
    
    def get_residual(self, measurement: float) -> float:
        return measurement - self.x

    def get_position(self) -> float:
        return self.x

    def get_velocity(self) -> float:
        return self.v
    
    def get_acceleration(self) -> float:
        return self.a

    def get_position_variance(self) -> float:
        """
        Повертає теоретичну стаціонарну дисперсію помилки позиції.
        Var(x) = K * R, де K залежить від alpha/beta.
        """
        if self.state_dim == 2:
            # Формула для сталої дисперсії alpha-beta фільтра
            # P_pos = [ (2*alpha^2 + 2*beta + alpha*beta) / (alpha * (4 - 2*alpha - beta)) ] * R
            # Спрощена версія:
            denominator = self.alpha * (4 - 2*self.alpha - self.beta)
            if abs(denominator) < 1e-6: denominator = 1e-6
            numerator = (2 * self.alpha**2 + 2 * self.beta + self.alpha * self.beta)
            k_gain = numerator / denominator
        else:
            # Для ABG це складніше, використовуємо наближення через alpha
            k_gain = self.alpha / (1 - self.alpha)
            
        return k_gain * self.R

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Прогноз на k кроків вперед"""
        preds = np.zeros(k)
        vars_pred = np.zeros(k)
        
        curr_x, curr_v, curr_a = self.x, self.v, self.a
        
        # Поточна невизначеність
        p0 = self.get_position_variance()
        
        for i in range(k):
            t = (i + 1) * self.dt
            if self.state_dim == 2:
                pred_x = curr_x + curr_v * t
                # Невизначеність зростає лінійно-квадратично
                # Var(t) ~ P0 + (t^2 * var_v)
                vars_pred[i] = p0 + (t**2) * (p0 * 0.1) # Approx growth
            else:
                pred_x = curr_x + curr_v * t + 0.5 * curr_a * t**2
                vars_pred[i] = p0 + (t**4) * (p0 * 0.01) # Approx growth
            
            preds[i] = pred_x
            
        return preds, vars_pred