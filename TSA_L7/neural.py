import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os

class DeepLearner:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.n_features = 3 # [Value, Sin_Time, Cos_Time]

    def _get_time_embeddings(self, dates):
        """Генерує циклічні ознаки часу."""
        day_of_year = dates.dayofyear.values
        sin_time = np.sin(2 * np.pi * day_of_year / 365.2425)
        cos_time = np.cos(2 * np.pi * day_of_year / 365.2425)
        return sin_time, cos_time

    def prepare_data(self, data_series: pd.Series, fit_scaler=True):
        values = data_series.values.reshape(-1, 1)
        if fit_scaler:
            scaled_values = self.scaler.fit_transform(values)
        else:
            scaled_values = self.scaler.transform(values)
            
        if not isinstance(data_series.index, pd.DatetimeIndex):
            print("  [WARN] Індекс не є датою. Генеруємо штучний календар.")
            dates = pd.date_range(start='2020-01-01', periods=len(data_series), freq='D')
        else:
            dates = data_series.index

        sin_time, cos_time = self._get_time_embeddings(dates)
        features = np.column_stack([scaled_values.flatten(), sin_time, cos_time])

        X, y = [], []
        for i in range(self.window_size, len(features)):
            X.append(features[i-self.window_size:i])
            y.append(scaled_values[i, 0])
            
        return np.array(X), np.array(y)

    def prepare_forecast_input(self, data_series):
        """Готує одне вікно даних для старту прогнозу (без y)."""
        values = data_series.values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        
        if not isinstance(data_series.index, pd.DatetimeIndex):
            dates = pd.date_range(start='2020-01-01', periods=len(data_series), freq='D')
        else:
            dates = data_series.index
            
        sin_time, cos_time = self._get_time_embeddings(dates)
        
        features = np.column_stack([scaled_values.flatten(), sin_time, cos_time])
        
        if len(features) < self.window_size:
            raise ValueError(f"Недостатньо даних для вікна! Треба {self.window_size}, є {len(features)}")
            
        return features[-self.window_size:].reshape(1, self.window_size, self.n_features)

    def build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.window_size, self.n_features)),
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.2),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        print(f"\n[NEURAL] Навчання LSTM (Embeddings Mode)...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        return self.history

    def predict(self, X_input):
        pred_scaled = self.model.predict(X_input, verbose=0)
        return self.scaler.inverse_transform(pred_scaled)
    
    def extrapolate(self, last_window_features, start_date, steps):
        current_batch = last_window_features.copy()
        predictions = []
        curr_date = start_date
        
        print(f"  [EXTRAPOLATE] Генерація {steps} кроків з урахуванням календаря...")
        
        for _ in range(steps):
            # Викликаємо модель напряму (як шар) для швидкості і щоб уникнути витоків пам'яті
            pred_tensor = self.model(current_batch, training=False)
            pred_val_scaled = pred_tensor.numpy()[0, 0]
            
            predictions.append(pred_val_scaled)
            
            # Наступна дата
            curr_date = curr_date + pd.Timedelta(days=1)
            day_of_year = curr_date.dayofyear
            
            next_sin = np.sin(2 * np.pi * day_of_year / 365.2425)
            next_cos = np.cos(2 * np.pi * day_of_year / 365.2425)
            
            # Новий вектор [Pred, Sin, Cos]
            next_step_features = np.array([[[pred_val_scaled, next_sin, next_cos]]], dtype=np.float32)
            
            # Зсув вікна
            current_batch = np.append(current_batch[:, 1:, :], next_step_features, axis=1)
            
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"  [SAVE] Модель збережено: {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"  [LOAD] Модель завантажено: {filepath}")
            return True
        return False