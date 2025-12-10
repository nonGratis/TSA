import numpy as np
from typing import Optional, Tuple

def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """Оцінка Q та R методом дисперсій різниць (без змін)."""
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
    Реалізація фільтра Калмана у матричному вигляді.
    Забезпечує точний розрахунок коваріації інновації (S) для NIS-аналізу.
    
    Зберігає інтерфейс AlphaBetaFilter для сумісності, але всередині працює 
    як повноцінний Kalman Filter.
    """
    def __init__(
        self,
        dt: float = 1.0,
        state_dim: int = 2,
        process_noise_q: float = 1.0, # Це спектральна щільність шуму (scalar scaling factor)
        measurement_noise_r: float = 1.0,
        init_state: Optional[np.ndarray] = None,
        alpha: Optional[float] = None # Deprecated в матричному режимі, ігнорується
    ):
        if state_dim not in [2, 3]:
            raise ValueError("state_dim must be 2 or 3")
        
        self.dt = dt
        self.state_dim = state_dim
        
        # 1. State Vector [x, v, (a)]
        self.x_state = np.zeros((state_dim, 1))
        if init_state is not None:
            self.x_state[0, 0] = float(init_state[0])
            if len(init_state) > 1: self.x_state[1, 0] = float(init_state[1])
            if len(init_state) > 2 and state_dim == 3: self.x_state[2, 0] = float(init_state[2])

        # 2. Matrices definition
        # Transition Matrix (A)
        if state_dim == 2:
            self.A = np.array([
                [1.0, dt],
                [0.0, 1.0]
            ])
            # Measurement Matrix (H) - we measure position only
            self.H = np.array([[1.0, 0.0]])
        else:
            self.A = np.array([
                [1.0, dt, 0.5*dt**2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0]
            ])
            self.H = np.array([[1.0, 0.0, 0.0]])

        # Identity Matrix
        self.I = np.eye(state_dim)

        # 3. Covariance Initialization (P)
        # Початкова невизначеність. Якщо невідомо, ставимо велику.
        self.P = np.eye(state_dim) * 100.0 
        
        # 4. Noise Matrices Initialization
        # Зберігаємо скалярні фактори для оновлень
        self.q_scalar = process_noise_q
        self.r_scalar = measurement_noise_r
        
        self.Q_mat = np.eye(state_dim) # Placeholder, буде оновлено в update_noise_matrices
        self.R_mat = np.array([[measurement_noise_r]])
        
        self.update_noise_matrices(process_noise_q, measurement_noise_r)
        
        # Public properties for access
        self.alpha = 0.0 # Will be synced with K[0]
        self.beta = 0.0
        self.S = 0.0 # Innovation covariance storage

    def update_noise_matrices(self, q: float, r: float) -> None:
        """Оновлення матриць шуму Q та R."""
        self.q_scalar = q
        self.r_scalar = max(r, 1e-9)
        self.R_mat[0, 0] = self.r_scalar
        
        # Формування матриці Q (Discrete White Noise Acceleration Model)
        # Q = q * [terms based on dt]
        if self.state_dim == 2:
            # Piecewise Constant White Acceleration approx for dim 2
            self.Q_mat = q * np.array([
                [0.25 * self.dt**4, 0.5 * self.dt**3],
                [0.5 * self.dt**3,      self.dt**2]
            ])
            # Альтернатива (спрощена, яку часто використовують):
            # self.Q_mat = q * np.array([[self.dt**3/3, self.dt**2/2], [self.dt**2/2, self.dt]])
        else:
            # Для dim=3 (Jerk as noise)
            dt = self.dt
            self.Q_mat = q * np.array([
                [dt**5/20, dt**4/8, dt**3/6],
                [dt**4/8,  dt**3/3, dt**2/2],
                [dt**3/6,  dt**2/2, dt]
            ])

    def update_params_from_noise(self, q: float, r: float) -> None:
        """Аліас для сумісності з пайплайном."""
        self.update_noise_matrices(q, r)

    def set_alpha(self, alpha: float) -> None:
        """
        [DEPRECATED] У матричному фільтрі Alpha вираховується автоматично через K.
        Цей метод залишено, щоб код не падав, але він не має ефекту на матрицю P.
        """
        pass

    def predict(self) -> float:
        # 1. State Extrapolation: x = A * x
        self.x_state = self.A @ self.x_state
        
        # 2. Covariance Extrapolation: P = A * P * A.T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q_mat
        
        return float(self.x_state[0, 0])

    def update(self, measurement: float, r_override: Optional[float] = None) -> float:
        """
        Крок оновлення (Update/Correction).
        
        Args:
            measurement: Виміряне значення.
            r_override: Опціонально - тимчасове значення R для цього кроку (для weighted update).
        """
        # Використовуємо тимчасовий R, якщо задано, інакше стандартний
        R_curr = np.array([[r_override]]) if r_override is not None else self.R_mat

        # 1. Innovation (Residual): y = z - H * x
        z = np.array([[measurement]])
        y = z - self.H @ self.x_state
        
        # 2. Innovation Covariance: S = H * P * H.T + R
        # Це єдиний математично коректний спосіб отримати S для NIS
        S_mat = self.H @ self.P @ self.H.T + R_curr
        self.S = float(S_mat[0, 0])
        
        # 3. Kalman Gain: K = P * H.T * S^-1
        K = self.P @ self.H.T / (self.S + 1e-12)
        
        # 4. State Update: x = x + K * y
        self.x_state = self.x_state + K @ y
        
        # 5. Covariance Update: P = (I - K * H) * P
        # Joseph form is more stable: P = (I-KH)P(I-KH)' + KRK', but simple form is usually enough here
        self.P = (self.I - K @ self.H) @ self.P
        
        # Sync simple properties
        self.alpha = float(K[0, 0])
        if self.state_dim >= 2:
            self.beta = float(K[1, 0]) * self.dt # Нормалізація beta до dt
        
        return float(self.x_state[0, 0])

    def get_residual(self, measurement: float) -> float:
        """Повертає інновацію (y) без оновлення стану."""
        z = np.array([[measurement]])
        y = z - self.H @ self.x_state
        return float(y[0, 0])

    def get_innovation_variance(self) -> float:
        """Повертає останнє розраховане (або передбачене) S."""
        # Якщо ми викликаємо це ДО update(), нам треба розрахувати S на основі P_pred
        S_mat = self.H @ self.P @ self.H.T + self.R_mat
        return float(S_mat[0, 0])
    
    # Гетери для сумісності
    def get_position(self) -> float: return float(self.x_state[0, 0])
    def get_velocity(self) -> float: return float(self.x_state[1, 0])
    def get_acceleration(self) -> float: 
        return float(self.x_state[2, 0]) if self.state_dim == 3 else 0.0

    def get_position_variance(self) -> float:
        """Повертає P[0,0] - дисперсію позиції."""
        return float(self.P[0, 0])
    
    # Властивості Q/R для пайплайну
    @property
    def Q(self) -> float: return self.q_scalar
    
    @property
    def R(self) -> float: return self.r_scalar

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Прогноз з використанням матриць."""
        preds = np.zeros(k)
        vars_pred = np.zeros(k)
        
        # Копіюємо стан для симуляції
        x_curr = self.x_state.copy()
        P_curr = self.P.copy()
        
        for i in range(k):
            # x = A x
            x_curr = self.A @ x_curr
            # P = A P A' + Q
            P_curr = self.A @ P_curr @ self.A.T + self.Q_mat
            
            preds[i] = float(x_curr[0, 0])
            vars_pred[i] = float(P_curr[0, 0])
                
        return preds, vars_pred