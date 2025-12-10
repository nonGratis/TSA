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
            # помірна невизначеність за замовчуванням
            self.P = np.eye(state_dim) * 10.0

    def _build_transition_matrix(self) -> np.ndarray:
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
        dt = self.dt
        if self.state_dim == 2:
            # Кінематична Q для [x, v] з безперервним білим шумом прискорення
            Q = q_scalar * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt]
            ])
        elif self.state_dim == 3:
            Q = q_scalar * np.array([
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0,  dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0,  dt**2 / 2.0, dt]
            ])
        else:
            Q = np.eye(self.state_dim) * q_scalar
        return Q

    def update_process_noise(self, q_scalar: float) -> None:
        self.Q = self._build_process_noise_matrix(q_scalar)

    def predict(self) -> np.ndarray:
        """
        Крок передбачення (predict step)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement: float) -> np.ndarray:
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def step(self, measurement: float) -> np.ndarray:
        self.predict()
        return self.update(measurement)

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> float:
        return float(self.x[0, 0])

    def get_velocity(self) -> float:
        return float(self.x[1, 0])

    def get_acceleration(self) -> Optional[float]:
        if self.state_dim >= 3:
            return float(self.x[2, 0])
        return None

    def get_residual(self, measurement: float) -> float:
        predicted_position = float(self.x[0, 0])
        return measurement - predicted_position

    def get_innovation_covariance(self) -> float:
        S = self.H @ self.P @ self.H.T + self.R
        return float(S[0, 0])

    def get_covariance_matrix(self) -> np.ndarray:
        return self.P.copy()

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогноз k кроків вперед без оновлення поточного стану.
        Повертає кортеж (predictions, position_variances), де position_variances
        — дисперсії P[0,0] для кожного прогнозного кроку.
        """
        x_backup = self.x.copy()
        P_backup = self.P.copy()

        predictions = np.zeros(k)
        pos_vars = np.zeros(k)

        for i in range(k):
            self.predict()
            predictions[i] = self.get_position()
            pos_vars[i] = float(self.P[0, 0])  # дисперсія позиції на крок i

        self.x = x_backup
        self.P = P_backup

        return predictions, pos_vars

    @staticmethod
    def compute_nis(residual: float, innovation_covariance: float) -> float:
        """
        Обчислити NIS (Normalized Innovation Squared) для одномірного виміру:
            NIS = residual^2 / S
        Додаємо захист від нульової S.
        """
        S = float(innovation_covariance)
        S_safe = max(S, 1e-12)
        return float((residual ** 2) / S_safe)

    def compute_nees_position(self, true_position: float) -> float:
        """
        Approximate NEES for position dimension (якщо відома істинна позиція).
        NEES_position = (x_est_post - x_true)^2 / P_post[0,0]

        Якщо істинна позиція невідома, можна передавати measurement як proxy,
        але це не ідентично справжньому NEES.
        """
        est_pos = self.get_position()
        e = est_pos - float(true_position)
        P = self.get_covariance_matrix()
        pos_var = float(P[0, 0])
        pos_var_safe = max(pos_var, 1e-12)
        return float((e ** 2) / pos_var_safe)
