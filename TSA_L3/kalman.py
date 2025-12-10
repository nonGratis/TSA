import numpy as np
from typing import Optional, Tuple

def estimate_noise_parameters(y: np.ndarray) -> Tuple[float, float]:
    """
    Robust Noise Estimation using IQR or nanvar to avoid outliers/NaNs.
    """
    # 1. Clean NaNs and Infs
    y_clean = y[np.isfinite(y)]
    
    if len(y_clean) < 5:
        return 1.0, 10.0

    # 2. First differences (Velocity approx)
    diffs = np.diff(y_clean)
    
    # 3. Robust Variance Estimation (using IQR to ignore outliers)
    # R approx = Variance of velocity (jitter)
    # Q approx = Variance of acceleration (jerk)
    
    def robust_var(data):
        if len(data) < 2: return 1.0
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        if iqr > 0:
            # Sigma approx = IQR / 1.35
            return (iqr / 1.35) ** 2
        else:
            return np.var(data) # Fallback to standard var

    measurement_noise = float(robust_var(diffs)) # R

    if len(y_clean) >= 5:
        second_diffs = np.diff(diffs)
        process_noise = float(robust_var(second_diffs)) # Q
    else:
        process_noise = measurement_noise / 10.0

    # Sanity bounds (щоб не було 0 або nan)
    process_noise = max(process_noise, 1e-3)
    measurement_noise = max(measurement_noise, 1e-2)

    return process_noise, measurement_noise


class AlphaBetaFilter:
    """
    Матричний фільтр Калмана (CV/CA model).
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
        
        self.dt = float(dt)
        self.state_dim = state_dim
        
        # Init State
        self.x_state = np.zeros((state_dim, 1))
        if init_state is not None:
            safe_init = np.nan_to_num(init_state, nan=0.0)
            for i in range(min(len(safe_init), state_dim)):
                self.x_state[i, 0] = float(safe_init[i])

        # Matrices
        if state_dim == 2: # Constant Velocity
            self.A = np.array([[1.0, dt], [0.0, 1.0]])
            self.H = np.array([[1.0, 0.0]])
        else: # Constant Acceleration
            self.A = np.array([[1.0, dt, 0.5*dt**2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
            self.H = np.array([[1.0, 0.0, 0.0]])

        self.I = np.eye(state_dim)
        
        # Initial Uncertainty
        self.P = np.eye(state_dim) * 1000.0 
        
        # Noise
        self.q_scalar = float(process_noise_q) if np.isfinite(process_noise_q) else 1.0
        self.r_scalar = float(measurement_noise_r) if np.isfinite(measurement_noise_r) else 10.0
        
        self.Q_mat = np.eye(state_dim)
        self.R_mat = np.array([[self.r_scalar]])
        
        self.update_noise_matrices(self.q_scalar, self.r_scalar)
        
        self.alpha = 0.0 
        self.S = 0.0 

    def update_noise_matrices(self, q: float, r: float) -> None:
        q = float(q) if np.isfinite(q) and q > 0 else 1e-6
        r = float(r) if np.isfinite(r) and r > 0 else 1e-6
        
        self.q_scalar = q
        self.r_scalar = r
        self.R_mat[0, 0] = r
        
        dt = self.dt
        if self.state_dim == 2:
            self.Q_mat = q * np.array([
                [0.25 * dt**4, 0.5 * dt**3],
                [0.5 * dt**3,      dt**2]
            ])
        else:
            self.Q_mat = q * np.array([
                [dt**5/20, dt**4/8, dt**3/6],
                [dt**4/8,  dt**3/3, dt**2/2],
                [dt**3/6,  dt**2/2, dt]
            ])

    def update_params_from_noise(self, q: float, r: float) -> None:
        self.update_noise_matrices(q, r)

    def set_alpha(self, alpha: float) -> None:
        pass # Not used in Matrix KF directly

    def predict(self) -> float:
        self.x_state = self.A @ self.x_state
        self.P = self.A @ self.P @ self.A.T + self.Q_mat
        return float(self.x_state[0, 0])

    def update(self, measurement: float, r_override: Optional[float] = None) -> float:
        if not np.isfinite(measurement):
            return float(self.x_state[0, 0])

        R_curr = np.array([[r_override]]) if r_override is not None else self.R_mat

        # Innovation
        z = np.array([[measurement]])
        y = z - self.H @ self.x_state
        
        # S = H P H' + R
        S_mat = self.H @ self.P @ self.H.T + R_curr
        self.S = float(S_mat[0, 0])
        
        # K = P H' S^-1
        K = self.P @ self.H.T / (self.S + 1e-12)
        
        # x = x + K y
        self.x_state = self.x_state + K @ y
        
        # P = (I - K H) P
        self.P = (self.I - K @ self.H) @ self.P
        
        self.alpha = float(K[0, 0])
        return float(self.x_state[0, 0])

    def get_residual(self, measurement: float) -> float:
        if not np.isfinite(measurement): return 0.0
        z = np.array([[measurement]])
        y = z - self.H @ self.x_state
        return float(y[0, 0])

    def get_innovation_variance(self) -> float:
        S_mat = self.H @ self.P @ self.H.T + self.R_mat
        return float(S_mat[0, 0])
    
    def get_position(self) -> float: return float(self.x_state[0, 0])
    def get_velocity(self) -> float: return float(self.x_state[1, 0])
    def get_acceleration(self) -> float: 
        return float(self.x_state[2, 0]) if self.state_dim == 3 else 0.0

    def get_position_variance(self) -> float:
        return float(self.P[0, 0])
    
    # Властивості Q/R для пайплайну
    @property
    def Q(self) -> float: return self.q_scalar
    @property
    def R(self) -> float: return self.r_scalar

    def predict_k_steps(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros(k)
        vars_pred = np.zeros(k)
        
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