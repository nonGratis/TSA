import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class DeepLearner:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def prepare_data(self, data_series, fit_scaler=True):
        """
        Підготовка даних для LSTM: (Samples, TimeSteps, Features)
        """
        values = data_series.values.reshape(-1, 1)
        
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(values)
        else:
            scaled_data = self.scaler.transform(values)

        X, y = [], []
        # Слайсинг вікном: беремо window_size точок як X, наступну як y
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        # Решейп для Keras: [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def build_lstm_model(self):
        """
        Конструювання Bidirectional LSTM (глибша та розумніша архітектура).
        """
        model = Sequential([
            Input(shape=(self.window_size, 1)),
            
            # Двосторонній шар: вчить контекст з минулого і "майбутнього" у вікні
            Bidirectional(LSTM(units=64, return_sequences=True)),
            Dropout(0.2), # Захист від перенавчання
            
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.2),
            
            Dense(units=32, activation='relu'),
            Dense(units=1) # Прогноз одного значення
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=64):
        print(f"\n[NEURAL] Навчання LSTM мережі ({epochs} епох)...")
        
        # Зупинка, якщо модель перестала вчитися (економія часу)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2, # 20% даних йде на валідацію під час навчання
            callbacks=[early_stop],
            verbose=1
        )
        return self.history

    def predict(self, X_input):
        """Прогноз на підготовлених даних + денормалізація."""
        pred_scaled = self.model.predict(X_input, verbose=0)
        return self.scaler.inverse_transform(pred_scaled)
    
    def extrapolate(self, last_window_scaled, steps):
        current_batch = last_window_scaled.reshape((1, self.window_size, 1))
        predictions = []
        
        print(f"  [EXTRAPOLATE] Генерація {steps} кроків...")
        for _ in range(steps):
            # Прогноз 1 кроку
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred)
            
            # Зсув вікна: викидаємо найстаріше, додаємо прогноз
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
            
        # Повертаємо до реального масштабу
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"  [SAVE] Модель збережено: {filepath}")

    def load_model(self, filepath):
        self.model = load_model(filepath)
        print(f"  [LOAD] Модель завантажено: {filepath}")