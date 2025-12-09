import numpy as np
from typing import Dict

class PolynomialModel:
    def __init__(self, y: np.ndarray):
        """
        Args:
            y: Вхідний часовий ряд
        """
        self.y = y
        self.n = len(y)
        self.X = np.arange(self.n)
        self.degree = 2
        self.coeffs = None
    
    def fit(self) -> None:
        """Fit degree-2 polynomial to the data"""
        self.coeffs = np.polyfit(self.X, self.y, self.degree)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given X using fitted polynomial
        
        Args:
            X: Input array (time indices)
            
        Returns:
            Predicted values
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.poly1d(self.coeffs)(X)
    
    def get_params(self) -> Dict:
        """Return model parameters"""
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return {'coeffs': self.coeffs.tolist(), 'degree': self.degree}
    
    def get_equation_string(self) -> str:
        """Return polynomial equation as string"""
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        a, b, c = self.coeffs
        return f"y(t) = {a:.4f}·t² + {b:.4f}·t + {c:.4f}"
    
    def generate_synthetic_data(self, distribution: str = 'normal') -> np.ndarray:
        """
        Generate synthetic data based on fitted trend and noise distribution
        
        Args:
            distribution: 'normal' or 'uniform'
            
        Returns:
            Synthetic data (trend + noise)
        """
        if self.coeffs is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        y_trend = self.predict(self.X)
        residuals = self.y - y_trend
        std = np.std(residuals, ddof=1)
        
        if distribution == 'normal':
            noise = np.random.normal(loc=0, scale=std, size=len(y_trend))
        elif distribution == 'uniform':
            delta = std * np.sqrt(3)
            noise = np.random.uniform(low=-delta, high=delta, size=len(y_trend))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return y_trend + noise
