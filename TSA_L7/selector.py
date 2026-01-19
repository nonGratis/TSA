import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from forecasting import ClassicalForecaster
from regression import RegressionForecaster
import metrics as mt

class ModelSelector:
    def __init__(self, data: pd.Series, freq_period: int = 24):
        self.data = data
        self.period = freq_period
        self.features = {}
        
    def analyze_properties(self):
        # 1. STL Декомпозиція
        decomp = seasonal_decompose(self.data, period=self.period, extrapolate_trend='freq')
        
        # 2. Обчислення сили компонент (F_trend, F_seasonal)
        # Формула: F = max(0, 1 - Var(Resid)/Var(Component + Resid))
        var_resid = np.var(decomp.resid)
        var_trend_resid = np.var(decomp.trend + decomp.resid)
        var_season_resid = np.var(decomp.seasonal + decomp.resid)
        
        trend_strength = max(0, 1 - var_resid / var_trend_resid)
        seasonal_strength = max(0, 1 - var_resid / var_season_resid)
        
        self.features = {
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength
        }
        return self.features

    def select_best_model(self, k_steps: int):
        props = self.analyze_properties()
        print(f"\n[AUTO-SELECTOR] Властивості: Trend={props['trend_strength']:.2f}, Seasonal={props['seasonal_strength']:.2f}")
        
        fc = ClassicalForecaster()
        rc = RegressionForecaster()
        
        train = self.data.iloc[:-k_steps]
        test = self.data.iloc[-k_steps:]
        
        candidates = {}
        recommendation = ""

        # ЛОГІКА ВИБОРУ (Recommendation Engine)
        if props['seasonal_strength'] > 0.4:
            recommendation = "Дані сезонні. Рекомендується: Holt-Winters."
            # Запускаємо Triple Exp Smoothing
            pred, _ = fc.holt_winters_forecast(train, steps=k_steps, seasonal_periods=self.period)
            candidates['Holt-Winters'] = pred
        
        elif props['trend_strength'] > 0.5:
            recommendation = "Дані трендові. Рекомендується: Holt (Double) або Regression."
            # Запускаємо регресії та Holt
            candidates['LinearReg'] = rc.linear_regression(train, steps=k_steps)
            candidates['PolyReg(2)'] = rc.polynomial_regression(train, steps=k_steps, degree=2)
            # Holt без сезонності (Double)
            # (Тут треба модифікувати fc.holt_winters_forecast щоб вимкнути seasonal, або додати окремий метод)
            pred_hw, _ = fc.holt_winters_forecast(train, steps=k_steps, seasonal_periods=None) 
            candidates['Holt-Double'] = pred_hw
            
        else:
            recommendation = "Дані без явного тренду/сезону. Рекомендується: Simple Exp або MA."
            candidates['MA'] = fc.moving_average_forecast(train, steps=k_steps)[0]
            # Simple Exp Smoothing
            # ...
            
        print(f"  Рекомендація: {recommendation}")
        
        # Фінальна валідація (Competition)
        best_name = None
        best_rmse = float('inf')
        
        print(f"\n  {'Model':<15} | {'RMSE':<10}")
        print("-" * 30)
        
        for name, pred in candidates.items():
            rmse = mt.calculate_rmse(test.values, pred)
            print(f"  {name:<15} | {rmse:<10.2f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                
        return best_name, candidates[best_name]