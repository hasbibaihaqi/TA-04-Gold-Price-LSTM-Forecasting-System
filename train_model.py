import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Dataset
print("[1/6] Membaca dataset...")
df = pd.read_csv('dataset_timeseries.csv')

# Bersihkan dan pastikan kolom Date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
elif 'Price' in df.columns and str(df['Price'].iloc[0]).lower() == 'ticker':
    # Kasus format raw multiindex yfinance
    df = pd.read_csv('dataset_timeseries.csv', skiprows=2)
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

df.sort_index(inplace=True)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(subset=['Close'], inplace=True)

print(f"Dataset berhasil dimuat: {len(df)} baris data.")
print(f"Rentang tanggal: {df.index.min().strftime('%Y-%m-%d')} s/d {df.index.max().strftime('%Y-%m-%d')}")

# 2. Scaling Data
print("\n[2/6] Normalisasi data menggunakan MinMaxScaler...")
close_prices = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 3. Windowing (Membuat Sekuens)
print("\n[3/6] Membuat sekuens windowing...")
TIME_STEP = 30

def create_sequences(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, time_step=TIME_STEP)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape ke Tensor 3D: (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Bentuk X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Bentuk X_test : {X_test.shape}, y_test : {y_test.shape}")

# 4. Membangun Model LSTM
print("\n[4/6] Membangun arsitektur Sequential LSTM...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(TIME_STEP, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 5. Pelatihan Model
print("\n[5/6] Melatih model (25 epochs)...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Evaluasi Model
print("\n[6/6] Evaluasi performa model...")
predictions = model.predict(X_test)

# Inverse Transform
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)

print("=" * 50)
print(f"HASIL EVALUASI MODEL LSTM:")
print(f" - Root Mean Squared Error (RMSE) : ${rmse:.2f}")
print(f" - Mean Absolute Error (MAE)     : ${mae:.2f}")
print("=" * 50)

# Simpan Artefak Model & Scaler
os.makedirs('model', exist_ok=True)
model.save('model/model_lstm.h5')
joblib.dump(scaler, 'model/scaler.pkl')

print("\n[SUKSES] Model dan scaler berhasil disimpan:")
print(" 1. model/model_lstm.h5")
print(" 2. model/scaler.pkl")

