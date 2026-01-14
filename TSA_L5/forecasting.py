import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ClassicalForecaster:
    """
    Реалізація класичних методів прогнозування:
    - Moving Average (MA)
    - Exponential Smoothing (Holt-Winters)
    - ARIMA
    """
    
    def __init__(self):
        self.models = {}
        
    def moving_average_forecast(self, data: pd.Series, window: int = 24, 
                              steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        # Розрахунок MA на історії
        ma = data.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        
        # Прогноз (наївний MA)
        # Для покращення можна взяти тренд зміни MA
        forecast = np.full(steps, last_ma)
        
        # Довірчі інтервали (базуються на std у вікні)
        last_std = data.rolling(window=window).std().iloc[-1]
        conf_int = np.column_stack([forecast - 1.96*last_std, forecast + 1.96*last_std])
        
        return forecast, conf_int

    def holt_winters_forecast(self, data: pd.Series, steps: int = 12, 
                            seasonal_periods: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Ініціалізація моделі (адитивна за замовчуванням)
            model = ExponentialSmoothing(
                data, 
                seasonal_periods=seasonal_periods,
                trend='add', 
                seasonal='add', 
                damped_trend=True,
                initialization_method="estimated"
            )
            fit = model.fit()
            

            forecast = fit.forecast(steps).values
            
            # Емуляція довірчих інтервалів (statsmodels для ETS це робить складно, беремо наближення)
            # Використовуємо залишкову дисперсію
            residuals = data - fit.fittedvalues
            std_resid = np.std(residuals)

            horizon_scale = np.sqrt(np.arange(1, steps + 1))
            lower = forecast - 1.96 * std_resid * horizon_scale
            upper = forecast + 1.96 * std_resid * horizon_scale
            conf_int = np.column_stack([lower, upper])
            
            self.models['hw'] = fit
            return forecast, conf_int
            
        except Exception as e:
            print(f"  [Error] Holt-Winters failed: {e}")
            return np.zeros(steps), np.zeros((steps, 2))

    def arima_forecast(self, data: pd.Series, order: Tuple[int,int,int] = (1,1,1), 
                      steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # Вимикаємо перевірку частоти для робастності
            model = ARIMA(data, order=order, enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit()
            
            forecast_res = fit.get_forecast(steps=steps)
            forecast = forecast_res.predicted_mean.values
            conf_int = forecast_res.conf_int(alpha=0.05).values
            
            self.models['arima'] = fit
            return forecast, conf_int
            
        except Exception as e:
            print(f"  [Error] ARIMA failed: {e}")
            return np.zeros(steps), np.zeros((steps, 2))

    def auto_arima_forecast(self, data: pd.Series, steps: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        best_aic = float('inf')
        best_order = (1, 1, 0)
        best_model = None
        
        # Обмежений перебір для швидкості
        p_values = [1, 2]
        d_values = [1] # Зазвичай 1 для нестаціонарних
        q_values = [0, 1, 2]
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(data, order=(p,d,q))
                        res = model.fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p,d,q)
                            best_model = res
                    except:
                        continue
        
        print(f"  [Auto-ARIMA] Best Order: {best_order} AIC:{best_aic:.1f}")
        
        if best_model:
            forecast_res = best_model.get_forecast(steps=steps)
            return forecast_res.predicted_mean.values, forecast_res.conf_int(alpha=0.05).values
        else:
            return self.arima_forecast(data, steps=steps)