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
        """Генерує циклічні ознаки часу (Sin/Cos року)."""
        day_of_year = dates.dayofyear.values
        # 365.2425 - середня довжина року
        sin_time = np.sin(2 * np.pi * day_of_year / 365.2425)
        cos_time = np.cos(2 * np.pi * day_of_year / 365.2425)
        return sin_time, cos_time

    def prepare_data(self, data_series: pd.Series, fit_scaler=True):
        """
        Підготовка даних з часовими ембедінгами.
        Вхід: Series з DatetimeIndex.
        Вихід: X=(Samples, Window, 3), y=(Samples,)
        """
        # 1. Масштабування значень (Target)
        values = data_series.values.reshape(-1, 1)
        if fit_scaler:
            scaled_values = self.scaler.fit_transform(values)
        else:
            scaled_values = self.scaler.transform(values)
            
        # 2. Часові ознаки (Time Embeddings)
        # Переконуємось, що індекс це дати
        if not isinstance(data_series.index, pd.DatetimeIndex):
            print("  [WARN] Індекс не є датою. Генеруємо штучний календар.")
            dates = pd.date_range(start='2020-01-01', periods=len(data_series), freq='D')
        else:
            dates = data_series.index

        sin_time, cos_time = self._get_time_embeddings(dates)
        
        # Об'єднуємо: [Value, Sin, Cos]
        features = np.column_stack([scaled_values.flatten(), sin_time, cos_time])

        X, y = [], []
        for i in range(self.window_size, len(features)):
            # X: вікно (всі 3 ознаки)
            X.append(features[i-self.window_size:i])
            # y: наступне значення (тільки Value)
            y.append(scaled_values[i, 0])
            
        return np.array(X), np.array(y)

    def build_lstm_model(self):
        """Архітектура з урахуванням 3-х вхідних фічей."""
        model = Sequential([
            # Input shape тепер (Window, 3)
            Input(shape=(self.window_size, self.n_features)),
            
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.2),
            
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.2),
            
            Dense(units=32, activation='relu'),
            Dense(units=1) # Вихід - скаляр
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
        """Прогноз на історії (Batch mode - тут predict ок)."""
        pred_scaled = self.model.predict(X_input, verbose=0)
        return self.scaler.inverse_transform(pred_scaled)
    
    def extrapolate(self, last_window_features, start_date, steps):
        """
        Рекурсивний прогноз з генерацією майбутніх дат.
        ВИПРАВЛЕНО: Виклик моделі напряму замість predict().
        """
        current_batch = last_window_features.copy() # (1, window, 3)
        predictions = []
        
        # Поточна дата для розрахунку майбутнього часу
        curr_date = start_date
        
        print(f"  [EXTRAPOLATE] Генерація {steps} кроків з урахуванням календаря...")
        
        for _ in range(steps):
            # 1. Прогноз значення (Value)
            # --- FIX START ---
            # Викликаємо модель як функцію (Callabale) для швидкості в циклі
            pred_tensor = self.model(current_batch, training=False)
            pred_val_scaled = pred_tensor.numpy()[0, 0]
            # --- FIX END ---
            
            predictions.append(pred_val_scaled)
            
            # 2. Обчислення часу для НАСТУПНОГО кроку
            curr_date = curr_date + pd.Timedelta(days=1)
            day_of_year = curr_date.dayofyear
            
            next_sin = np.sin(2 * np.pi * day_of_year / 365.2425)
            next_cos = np.cos(2 * np.pi * day_of_year / 365.2425)
            
            # 3. Формування нового вектора входу [Pred_Value, Next_Sin, Next_Cos]
            # Важливо зберегти розмірності (1, 1, 3) для конкатенації
            next_step_features = np.array([[[pred_val_scaled, next_sin, next_cos]]], dtype=np.float32)
            
            # 4. Зсув вікна: викидаємо старе (0-й елемент), додаємо нове в кінець
            current_batch = np.append(current_batch[:, 1:, :], next_step_features, axis=1)
            
        # Денормалізація тільки значень
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