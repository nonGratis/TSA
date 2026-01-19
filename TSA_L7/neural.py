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
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.history = None
        self.n_features = 1

    def prepare_data(self, data_series: pd.Series, fit_scaler=True):

        values = data_series.values.reshape(-1, 1)
        
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(values)
        else:
            scaled_data = self.scaler.transform(values)

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], self.n_features))
        
        return X, y

    def prepare_forecast_input(self, data_series):
        """Готує одне вікно даних для старту (shape: 1, window, 1)."""
        values = data_series.values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        
        if len(scaled_values) < self.window_size:
             raise ValueError(f"Мало даних: треба {self.window_size}, є {len(scaled_values)}")
             
        # Беремо останні window точок
        return scaled_values[-self.window_size:].reshape(1, self.window_size, self.n_features)

    def build_lstm_model(self):
        """Класична Bidirectional LSTM."""
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
        print(f"\n[NEURAL] Навчання LSTM (Residuals Mode)...")
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
    
    def extrapolate(self, last_window_scaled, steps):
        current_batch = last_window_scaled.copy() # (1, window, 1)
        predictions = []
        
        print(f"  [EXTRAPOLATE] Генерація {steps} кроків залишків...")
        
        for _ in range(steps):
            # Викликаємо як функцію для швидкості
            pred_tensor = self.model(current_batch, training=False)
            pred_val_scaled = pred_tensor.numpy()[0, 0]
            
            predictions.append(pred_val_scaled)
            
            # Зсув вікна
            next_step = np.array([[[pred_val_scaled]]], dtype=np.float32)
            current_batch = np.append(current_batch[:, 1:, :], next_step, axis=1)
            
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