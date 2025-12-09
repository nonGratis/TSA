import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path

class ModelComparator:
    """
    Compare polynomial and logistic regression models with statistical metrics.
    
    Metrics:
        RMSE: Root Mean Squared Error
        MAE: Mean Absolute Error
        R²: Coefficient of Determination
        AIC: Akaike Information Criterion
        BIC: Bayesian Information Criterion
    """
    
    def __init__(self):
        """Initialize comparator"""
        self.results = {}
    
    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of Determination (R²)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    @staticmethod
    def _calculate_aic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Akaike Information Criterion
        
        AIC = 2k + n·ln(RSS/n)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            k: Number of parameters
        """
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        
        if rss <= 0:
            return np.inf
        
        return 2 * k + n * np.log(rss / n)
    
    @staticmethod
    def _calculate_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Bayesian Information Criterion
        
        BIC = k·ln(n) + n·ln(RSS/n)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            k: Number of parameters
        """
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        
        if rss <= 0:
            return np.inf
        
        return k * np.log(n) + n * np.log(rss / n)
    
    def evaluate_model(
        self, 
        model_name: str, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_params: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single model with all metrics
        
        Args:
            model_name: Name of the model ('Polynomial' or 'Logistic')
            y_true: Actual values
            y_pred: Predicted values
            n_params: Number of model parameters
            params: Model parameters (for reporting)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'model': model_name,
            'RMSE': self._calculate_rmse(y_true, y_pred),
            'MAE': self._calculate_mae(y_true, y_pred),
            'R²': self._calculate_r2(y_true, y_pred),
            'AIC': self._calculate_aic(y_true, y_pred, n_params),
            'BIC': self._calculate_bic(y_true, y_pred, n_params),
            'n_params': n_params,
            'params': params
        }
        
        self.results[model_name] = metrics
        return metrics
    
    @staticmethod
    def compare(poly_metrics: Dict, logistic_metrics: Dict) -> Dict[str, Any]:
        """
        Compare two models and determine the winner
        
        Args:
            poly_metrics: Polynomial model metrics
            logistic_metrics: Logistic model metrics
            
        Returns:
            Comparison summary with winners for each metric
        """
        comparison: Dict[str, Any] = {
            'polynomial': poly_metrics,
            'logistic': logistic_metrics,
            'winners': {}
        }
        
        # Lower is better for RMSE, MAE, AIC, BIC
        for metric in ['RMSE', 'MAE', 'AIC', 'BIC']:
            poly_val = poly_metrics[metric]
            log_val = logistic_metrics[metric]
            
            if poly_val < log_val:
                comparison['winners'][metric] = 'Поліном'
            elif log_val < poly_val:
                comparison['winners'][metric] = 'Логістична'
            else:
                comparison['winners'][metric] = 'Нічия'
        
        # Higher is better for R²
        poly_r2 = poly_metrics['R²']
        log_r2 = logistic_metrics['R²']
        
        if poly_r2 > log_r2:
            comparison['winners']['R²'] = 'Поліном'
        elif log_r2 > poly_r2:
            comparison['winners']['R²'] = 'Логістична'
        else:
            comparison['winners']['R²'] = 'Нічия'
        
        # Overall winner (based on AIC - preferred information criterion)
        if poly_metrics['AIC'] < logistic_metrics['AIC']:
            comparison['overall_winner'] = 'Поліном'
        elif logistic_metrics['AIC'] < poly_metrics['AIC']:
            comparison['overall_winner'] = 'Логістична'
        else:
            comparison['overall_winner'] = 'Нічия'
        
        return comparison
    
    def export_csv(self, filepath: str, comparison: Optional[Dict] = None) -> None:
        """
        Export comparison results to CSV
        
        Args:
            filepath: Path to save CSV file
            comparison: Comparison dictionary (if None, uses self.results)
        """
        if comparison is None:
            if not self.results:
                raise ValueError("No results to export. Run evaluate_model first.")
            rows = list(self.results.values())
        else:
            rows = [comparison['polynomial'], comparison['logistic']]
        
        # Create DataFrame
        df_data = []
        for row in rows:
            df_data.append({
                'Model': row['model'],
                'RMSE': row['RMSE'],
                'MAE': row['MAE'],
                'R²': row['R²'],
                'AIC': row['AIC'],
                'BIC': row['BIC'],
                'Parameters': row['n_params'],
                'Params_Details': str(row['params'])
            })
        
        df = pd.DataFrame(df_data)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Результати порівняння збережено в: {filepath}")
    
    @staticmethod
    def print_summary(comparison: Dict) -> None:
        """
        Print formatted comparison summary to console
        
        Args:
            comparison: Comparison dictionary from compare()
        """
        print("\nРЕЗЮМЕ ПОРІВНЯННЯ МОДЕЛЕЙ")
        
        poly = comparison['polynomial']
        logistic = comparison['logistic']
        winners = comparison['winners']
        
        print(f"\n{'Метрика':<10} | {'Поліном':<15} | {'Логістична':<15} | {'Переможець':<15}")
        
        for metric in ['RMSE', 'MAE', 'R²', 'AIC', 'BIC']:
            poly_val = poly[metric]
            log_val = logistic[metric]
            winner = winners[metric]
            
            print(f"{metric:<10} | {poly_val:<15.4f} | {log_val:<15.4f} | {winner:<15}")
        
        print(f"\nЗАГАЛЬНИЙ ПЕРЕМОЖЕЦЬ (за AIC): {comparison['overall_winner']}")
