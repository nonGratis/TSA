import numpy as np
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

    def prepare_data(self, data_series, fit_scaler=True):
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
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_lstm_model(self):
        model = Sequential([
            Input(shape=(self.window_size, 1)),
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
        print(f"\n[NEURAL] Навчання LSTM мережі ({epochs} епох)...")
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
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
        current_batch = last_window_scaled.reshape((1, self.window_size, 1))
        predictions = []
        for _ in range(steps):
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"  [SAVE] Модель успішно збережено у: {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"  [LOAD] Модель завантажено з файлу: {filepath}")
            return True
        return False