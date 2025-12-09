from typing import Dict
import numpy as np


class LogisticRND:
    """
    Logistic regression model with manual gradient descent (Adam optimizer).
    Implements S-curve fitting: y(t) = L / (1 + e^(-k(t - t0)))
    
    Параметри:
        L (Capacity): Максимальна можлива кількість реакцій (асимптота)
        k (Growth Rate): Крутизна зростання (швидкість віральності)
        t0 (Midpoint): Час максимальної швидкості приросту (пік хайпу)
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 5000, optimizer: str = 'adam'):
        """
        Args:
            learning_rate: Швидкість навчання для градієнтного спуску
            epochs: Кількість ітерацій навчання
            optimizer: Тип оптимізатора ('adam' або 'sgd')
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.params: Dict[str, float] = {'L': 0.0, 'k': 0.0, 't0': 0.0}
        
        # Для Adam optimizer (історія градієнтів)
        self.m = {'L': 0.0, 'k': 0.0, 't0': 0.0}  # First moment
        self.v = {'L': 0.0, 'k': 0.0, 't0': 0.0}  # Second moment
        self.t_iter = 0
        
        # Параметри масштабування
        self.t_min = 0.0
        self.t_scale = 1.0
        self.y_scale = 1.0
        
        # Історія навчання
        self.loss_history = []
    
    def _stable_sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Числово стабільна сигмоїда для уникнення overflow у exp()
        
        σ(z) = 1/(1+e^(-z)) для z >= 0
        σ(z) = e^z/(1+e^z) для z < 0
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _initialize_params(self, t: np.ndarray, y: np.ndarray) -> None:
        """
        Розумна ініціалізація параметрів на основі даних
        
        L: 1.05 × max(y) - трохи вище максимуму для запасу
        t0: час, де y досягає половини максимуму
        k: оцінка нахилу в точці t0
        """
        # L: трохи вище максимуму
        self.params['L'] = 1.05 * np.max(y)
        
        # t0: точка де y ≈ L/2
        idx = np.abs(y - self.params['L'] / 2).argmin()
        self.params['t0'] = float(t[idx])
        
        # k: оцінка з похідної в точці t0
        if 0 < idx < len(t) - 1:
            dy = y[idx + 1] - y[idx - 1]
            dt = t[idx + 1] - t[idx - 1]
            slope = dy / dt if dt != 0 else 1.0
        else:
            slope = 1.0
        
        # k = 4 * slope / L (з властивостей похідної логістичної функції)
        self.params['k'] = 4 * slope / self.params['L']
        
        # Гарантуємо позитивність k
        self.params['k'] = max(self.params['k'], 0.1)
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Обчислення MSE (Mean Squared Error)"""
        return float(np.mean((y_true - y_pred) ** 2))
    
    def fit(self, t_raw: np.ndarray, y_raw: np.ndarray, verbose: bool = False) -> None:
        """
        Навчання моделі методом градієнтного спуску з Adam оптимізацією
        
        Args:
            t_raw: Часові мітки (numpy array)
            y_raw: Значення (numpy array)
            verbose: Чи виводити прогрес навчання
        """
        # 1. Масштабування для стабільності
        self.t_min = np.min(t_raw)
        self.t_scale = np.max(t_raw) - self.t_min
        if self.t_scale == 0:
            self.t_scale = 1.0
            
        self.y_scale = np.max(y_raw)
        if self.y_scale == 0:
            self.y_scale = 1.0
        
        t = (t_raw - self.t_min) / self.t_scale
        y = y_raw / self.y_scale
        
        # 2. Ініціалізація параметрів
        self._initialize_params(t, y)
        
        # 3. Основний цикл навчання
        for epoch in range(self.epochs):
            self.t_iter += 1
            
            # Forward pass
            L, k, t0 = self.params['L'], self.params['k'], self.params['t0']
            
            # Обчислення передбачень з clip для стабільності
            z = -k * (t - t0)
            z = np.clip(z, -500, 500)
            exp_term = np.exp(z)
            y_pred = L / (1 + exp_term)
            
            # Обчислення помилки
            error = y - y_pred
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Виведення прогресу
            if verbose and (epoch % 500 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.6f}, L={L*self.y_scale:.2f}, k={k:.4f}, t0={t0:.4f}")
            
            # --- Ручний розрахунок градієнтів (Chain Rule) ---
            # ∂J/∂L
            dL = -2 * np.mean(error * (1 / (1 + exp_term)))
            
            # Спільна частина для dk та dt0
            common_term = (L * exp_term) / ((1 + exp_term) ** 2)
            
            # ∂J/∂k
            dk = -2 * np.mean(error * common_term * (t - t0))
            
            # ∂J/∂t0
            dt0 = -2 * np.mean(error * common_term * (-k))
            
            gradients = {'L': dL, 'k': dk, 't0': dt0}
            
            # Оновлення параметрів
            self._update_params(gradients)
        
        # 4. Повертаємо L до реального масштабу
        self.params['L'] *= self.y_scale
    
    def _update_params(self, grads: Dict[str, float]) -> None:
        """
        Оновлення параметрів за допомогою Adam optimizer
        
        Adam = Adaptive Moment Estimation
        Комбінує Momentum (перший момент) та RMSProp (другий момент)
        """
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for key in self.params:
            g = grads[key]
            
            # Momentum (перший момент)
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            
            # RMSProp (другий момент)
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - beta1 ** self.t_iter)
            v_hat = self.v[key] / (1 - beta2 ** self.t_iter)
            
            # Оновлення параметра
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def predict(self, t_raw: np.ndarray) -> np.ndarray:
        """
        Прогнозування значень для заданих часових міток
        
        Args:
            t_raw: Часові мітки для прогнозування
            
        Returns:
            Передбачені значення
        """
        # Масштабування вхідного часу
        t = (t_raw - self.t_min) / self.t_scale
        
        # Розрахунок (L вже в реальному масштабі після fit)
        L = self.params['L']  # Real scale
        k = self.params['k']  # Normalized scale
        t0 = self.params['t0']  # Normalized scale
        
        z = -k * (t - t0)
        z = np.clip(z, -500, 500)
        
        return L / (1 + np.exp(z))
    
    def get_params(self) -> Dict[str, float]:
        """Повернути поточні параметри моделі"""
        return self.params.copy()
    
    def get_equation_string(self) -> str:
        """Повернути рівняння моделі у текстовому форматі"""
        L, k, t0 = self.params['L'], self.params['k'], self.params['t0']
        return f"y(t) = {L:.2f} / (1 + exp(-{k:.4f} * (t - {t0:.2f})))"
