import numpy as np
from typing import Optional, Tuple


def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """
    Оцінка параметрів шуму на основі даних
    
    Args:
        y: Часовий ряд спостережень
        
    Returns:
        (process_noise, measurement_noise): Оцінка Q та R
    """
    if len(y) < 2:
        return 1.0, 1.0
    
    # Measurement noise (R) - дисперсія перших різниць (а не всього ряду)
    # Це дає реалістичну оцінку шуму вимірювань
    diffs = np.diff(y)
    measurement_noise = float(np.var(diffs, ddof=1))
    
    # Process noise (Q) - оцінка з дисперсії других різниць (прискорення)
    if len(y) >= 3:
        second_diffs = np.diff(diffs)
        process_noise = float(np.var(second_diffs, ddof=1)) / 2.0
    else:
        process_noise = measurement_noise / 10.0
    
    # Мінімальні значення для стабільності
    process_noise = max(process_noise, 1e-6)
    measurement_noise = max(measurement_noise, 1e-6)
    
    return process_noise, measurement_noise


class KalmanFilter:
    """
    Матричний Kalman-фільтр для кумулятивних часових рядів
    
    Підтримує два режими:
    - state_dim=2: стан [x, v] (позиція, швидкість)
    - state_dim=3: стан [x, v, a] (позиція, швидкість, прискорення)
    
    Матриці:
    - F: Матриця переходу стану
    - H: Матриця спостереження
    - Q: Коваріація процесного шуму
    - R: Коваріація шуму вимірювання
    - P: Коваріація оцінки стану
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        state_dim: int = 2,
        process_noise_q: Optional[float] = None,
        measurement_noise_r: Optional[float] = None,
        init_state: Optional[np.ndarray] = None,
        init_P: Optional[np.ndarray] = None
    ):
        """
        Args:
            dt: Крок дискретизації (годинний інтервал = 1.0)
            state_dim: Розмірність стану (2 або 3)
            process_noise_q: Скаляр процесного шуму (якщо None - автоініціалізація)
            measurement_noise_r: Скаляр шуму вимірювання
            init_state: Початковий стан [x, v] або [x, v, a]
            init_P: Початкова коваріація оцінки
        """
        if state_dim not in [2, 3]:
            raise ValueError("state_dim має бути 2 або 3")
        
        self.dt = dt
        self.state_dim = state_dim
        
        # Ініціалізація стану
        if init_state is not None:
            self.x = np.array(init_state, dtype=float).reshape(-1, 1)
        else:
            self.x = np.zeros((state_dim, 1))
        
        # Матриця переходу F
        self.F = self._build_transition_matrix()
        
        # Матриця спостереження H (вимірюємо тільки позицію x)
        self.H = np.zeros((1, state_dim))
        self.H[0, 0] = 1.0
        
        # Коваріація процесного шуму Q
        if process_noise_q is None:
            process_noise_q = 1.0
        self.Q = self._build_process_noise_matrix(process_noise_q)
        
        # Коваріація шуму вимірювання R
        if measurement_noise_r is None:
            measurement_noise_r = 1.0
        self.R = np.array([[measurement_noise_r]])
        
        # Коваріація оцінки P
        if init_P is not None:
            self.P = np.array(init_P, dtype=float)
        else:
            self.P = np.eye(state_dim) * 1000.0  # Велика невизначеність на початку
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Побудова матриці переходу стану F
        
        Для state_dim=2:
            F = [[1, dt],
                 [0,  1]]
        
        Для state_dim=3:
            F = [[1, dt, 0.5*dt²],
                 [0,  1,     dt   ],
                 [0,  0,      1   ]]
        """
        if self.state_dim == 2:
            F = np.array([
                [1.0, self.dt],
                [0.0, 1.0]
            ])
        else:  # state_dim == 3
            dt2 = 0.5 * self.dt * self.dt
            F = np.array([
                [1.0, self.dt, dt2],
                [0.0, 1.0, self.dt],
                [0.0, 0.0, 1.0]
            ])
        
        return F
    
    def _build_process_noise_matrix(self, q_scalar: float) -> np.ndarray:
        """
        Побудова матриці процесного шуму Q
        
        Використовує кінематичну модель з корельованими компонентами
        
        Args:
            q_scalar: Скаляр для масштабування
            
        Returns:
            Матриця Q розміром (state_dim, state_dim)
        """
        dt = self.dt
        
        if self.state_dim == 2:
            # Кінематична Q для [x, v] з безперервним білим шумом прискорення
            Q = q_scalar * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt]
            ])
        elif self.state_dim == 3:
            # Кінематична Q для [x, v, a] з безперервним білим шумом jerk
            Q = q_scalar * np.array([
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0,  dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0,  dt**2 / 2.0, dt]
            ])
        else:
            # Fallback до діагональної матриці
            Q = np.eye(self.state_dim) * q_scalar
        
        return Q
    
    def update_process_noise(self, q_scalar: float) -> None:
        """
        Оновлення процесного шуму Q (для адаптації)
        
        Args:
            q_scalar: Новий скаляр процесного шуму
        """
        self.Q = self._build_process_noise_matrix(q_scalar)
    
    def predict(self) -> np.ndarray:
        """
        Крок передбачення (predict step)
        
        x̂_k|k-1 = F × x̂_k-1|k-1
        P_k|k-1 = F × P_k-1|k-1 × F^T + Q
        
        Returns:
            Передбачений стан
        """
        # Передбачення стану
        self.x = self.F @ self.x
        
        # Передбачення коваріації
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement: float) -> np.ndarray:
        """
        Крок оновлення (update step)
        
        y_k = z_k - H × x̂_k|k-1  (інновація/residual)
        S_k = H × P_k|k-1 × H^T + R  (коваріація інновації)
        K_k = P_k|k-1 × H^T × S_k^-1  (Kalman gain)
        x̂_k|k = x̂_k|k-1 + K_k × y_k
        P_k|k = (I - K_k × H) × P_k|k-1
        
        Args:
            measurement: Виміряне значення
            
        Returns:
            Оновлений стан
        """
        z = np.array([[measurement]])
        
        # Інновація (residual)
        y = z - self.H @ self.x
        
        # Коваріація інновації
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Оновлення стану
        self.x = self.x + K @ y
        
        # Оновлення коваріації (використовуємо Joseph form для числової стабільності)
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def step(self, measurement: float) -> np.ndarray:
        """
        Повний крок Kalman-фільтра: predict + update
        
        Args:
            measurement: Виміряне значення
            
        Returns:
            Оновлений стан після predict + update
        """
        self.predict()
        return self.update(measurement)
    
    def get_state(self) -> np.ndarray:
        """Повернути поточний стан"""
        return self.x.copy()
    
    def get_position(self) -> float:
        """Повернути оцінку позиції (перший елемент стану)"""
        return float(self.x[0, 0])
    
    def get_velocity(self) -> float:
        """Повернути оцінку швидкості (другий елемент стану)"""
        return float(self.x[1, 0])
    
    def get_acceleration(self) -> Optional[float]:
        """Повернути оцінку прискорення (якщо state_dim=3)"""
        if self.state_dim >= 3:
            return float(self.x[2, 0])
        return None
    
    def get_residual(self, measurement: float) -> float:
        """
        Обчислити залишок (residual) між вимірюванням та передбаченням
        
        Args:
            measurement: Виміряне значення
            
        Returns:
            Залишок (measurement - predicted_position)
        """
        predicted_position = float(self.x[0, 0])
        return measurement - predicted_position
    
    def get_innovation_covariance(self) -> float:
        """
        Обчислити інноваційну коваріацію S = H P H^T + R
        
        Returns:
            Скаляр S (для одновимірного вимірювання)
        """
        S = self.H @ self.P @ self.H.T + self.R
        return float(S[0, 0])
    
    def predict_k_steps(self, k: int) -> np.ndarray:
        """
        Прогноз k кроків вперед без оновлення поточного стану
        
        Args:
            k: Кількість кроків для прогнозу
            
        Returns:
            Масив передбачених позицій розміром k
        """
        # Зберігаємо поточний стан
        x_backup = self.x.copy()
        P_backup = self.P.copy()
        
        predictions = np.zeros(k)
        
        for i in range(k):
            self.predict()
            predictions[i] = self.get_position()
        
        # Відновлюємо стан
        self.x = x_backup
        self.P = P_backup
        
        return predictions
