import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, render_template, jsonify, request
from download_data import download_gold_data

app = Flask(__name__)

# Path file artefak dan dataset
MODEL_PATH = os.path.join('model', 'model_lstm.h5')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
EVAL_JSON_PATH = os.path.join('model', 'evaluation.json')
DATASET_PATH = 'dataset_timeseries.csv'
TIME_STEP = 30

# Memuat model LSTM dan scaler MinMaxScaler saat server dinyalakan
print("[INFO] Memuat model LSTM dan scaler...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Model dan scaler berhasil dimuat!")
except Exception as e:
    print(f"[WARNING] Gagal memuat model/scaler: {e}")
    model = None
    scaler = None


def load_dataset():
    """Membaca dataset CSV dan memastikan urutan kronologis berdasarkan tanggal."""
    df = pd.read_csv(DATASET_PATH)
    
    # Deteksi kolom tanggal secara fleksibel
    date_cols = [col for col in df.columns if 'date' in str(col).lower()]
    date_col = date_cols[0] if date_cols else df.columns[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, date_col


# Cache evaluasi agar pemanggilan cepat
EVAL_CACHE = None

def compute_evaluation():
    """Menghitung atau memuat metrik performa (RMSE, MAE, MAPE) pada data uji."""
    global EVAL_CACHE
    if EVAL_CACHE is not None:
        return EVAL_CACHE
    
    # 1. Coba baca dari file cache evaluasi json yang sudah dihitung (sangat cepat & hemat RAM)
    if os.path.exists(EVAL_JSON_PATH):
        try:
            with open(EVAL_JSON_PATH, 'r') as f:
                EVAL_CACHE = json.load(f)
                return EVAL_CACHE
        except Exception as e:
            print(f"[WARNING] Gagal membaca evaluation.json: {e}")

    if model is None or scaler is None:
        return None

    df, date_col = load_dataset()
    close_prices = df[['Close']].values
    scaled = scaler.transform(close_prices)

    # Buat sekuens
    X, y = [], []
    for i in range(len(scaled) - TIME_STEP):
        X.append(scaled[i:(i + TIME_STEP), 0])
        y.append(scaled[i + TIME_STEP, 0])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_test = X[train_size:].reshape(-1, TIME_STEP, 1)
    y_test = y[train_size:]

    # Prediksi data uji menggunakan model call langsung (eager)
    pred_scaled = model(X_test, training=False).numpy()
    pred_actual = scaler.inverse_transform(pred_scaled).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Hitung metrik
    rmse = float(np.sqrt(mean_squared_error(y_test_actual, pred_actual)))
    mae = float(mean_absolute_error(y_test_actual, pred_actual))
    mape = float(np.mean(np.abs((y_test_actual - pred_actual) / y_test_actual)) * 100)
    accuracy = round(100.0 - mape, 2)

    # Ambil 60 observasi terakhir data uji untuk grafik perbandingan
    dates_test = df[date_col].iloc[train_size + TIME_STEP:].dt.strftime('%Y-%m-%d').tolist()
    sample_len = min(60, len(dates_test))

    EVAL_CACHE = {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'mape': round(mape, 2),
        'accuracy': accuracy,
        'total_samples': len(df),
        'train_samples': train_size,
        'test_samples': len(X_test),
        'labels': dates_test[-sample_len:],
        'actual': [round(float(v), 2) for v in y_test_actual[-sample_len:]],
        'predicted': [round(float(v), 2) for v in pred_actual[-sample_len:]]
    }
    return EVAL_CACHE


@app.route('/')
def index():
    """Halaman utama dasbor web."""
    return render_template('index.html')


@app.route('/api/historical', methods=['GET'])
def get_historical():
    """Endpoint API untuk menyajikan data historis 60 hari terakhir ke Chart.js."""
    try:
        df, date_col = load_dataset()
        recent_df = df.tail(60)

        labels = recent_df[date_col].dt.strftime('%Y-%m-%d').tolist()
        prices = [round(float(p), 2) for p in recent_df['Close']]

        return jsonify({
            'status': 'success',
            'labels': labels,
            'prices': prices,
            'stats': {
                'min_price': round(float(recent_df['Close'].min()), 2),
                'max_price': round(float(recent_df['Close'].max()), 2),
                'avg_price': round(float(recent_df['Close'].mean()), 2)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    """Endpoint API inferensi untuk memprediksi harga penutupan emas (1 hingga 7 hari ke depan)."""
    try:
        if model is None or scaler is None:
            return jsonify({
                'status': 'error',
                'message': 'Model atau scaler belum tersedia di folder model/. Silakan latih model terlebih dahulu.'
            }), 500

        days = int(request.args.get('days', 1))
        days = max(1, min(days, 14)) # batasi 1 - 14 hari

        df, date_col = load_dataset()

        # Ambil 30 data penutupan terakhir
        last_30_days = df['Close'].tail(TIME_STEP).values.tolist()
        last_date = df[date_col].iloc[-1]
        last_price = float(df['Close'].iloc[-1])

        # Buffer harga untuk multi-step forecasting
        current_window = np.array(last_30_days).reshape(-1, 1)
        current_scaled = scaler.transform(current_window).flatten().tolist()

        forecast_results = []
        curr_date = last_date

        for step in range(days):
            # Input 30 harga terakhir
            x_input = np.array(current_scaled[-TIME_STEP:], dtype=np.float32).reshape(1, TIME_STEP, 1)
            pred_scaled_val = float(model(x_input, training=False).numpy()[0][0])
            
            # Inverse transform
            pred_price_val = float(scaler.inverse_transform([[pred_scaled_val]])[0][0])
            
            # Hitung tanggal hari perdagangan berikutnya (tambah 1 hari)
            curr_date = curr_date + pd.Timedelta(days=1)
            
            # Tambahkan ke buffer untuk iterasi hari berikutnya
            current_scaled.append(pred_scaled_val)

            forecast_results.append({
                'step': step + 1,
                'date': curr_date.strftime('%Y-%m-%d'),
                'predicted_price': round(pred_price_val, 2)
            })

        # Prediksi 1 hari ke depan (kompatibilitas backward)
        first_pred = forecast_results[0]
        price_change = first_pred['predicted_price'] - last_price
        percentage_change = (price_change / last_price) * 100

        return jsonify({
            'status': 'success',
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_price': round(last_price, 2),
            'next_date': first_pred['date'],
            'predicted_price': first_pred['predicted_price'],
            'price_change': round(price_change, 2),
            'percentage_change': round(percentage_change, 2),
            'trend': 'UP' if price_change >= 0 else 'DOWN',
            'forecast': forecast_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    """Endpoint API menyajikan metrik performa model (RMSE, MAE) dan data uji vs prediksi."""
    try:
        eval_data = compute_evaluation()
        if eval_data is None:
            return jsonify({'status': 'error', 'message': 'Model tidak siap'}), 500
        return jsonify({
            'status': 'success',
            'data': eval_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/sync-data', methods=['POST', 'GET'])
def sync_data():
    """Endpoint untuk memperbarui dataset langsung dari Yahoo Finance."""
    global EVAL_CACHE
    try:
        success, info = download_gold_data()
        if not success:
            return jsonify({'status': 'error', 'message': str(info)}), 500
        
        # Reset cache evaluasi agar metrik dan grafik menggunakan dataset terbaru
        EVAL_CACHE = None
        
        return jsonify({
            'status': 'success',
            'message': f"Sinkronisasi berhasil! Total {info['total_rows']} baris data. Tanggal pasar terbaru: {info['latest_date']}.",
            'data': info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/simulate', methods=['GET', 'POST'])
def simulate():
    """Endpoint Simulator: Menghitung prediksi hari berikutnya berdasarkan skenario harga kustom hari ini."""
    try:
        if model is None or scaler is None:
            return jsonify({'status': 'error', 'message': 'Model tidak siap'}), 500

        # Ambil harga kustom dari request
        custom_price = None
        if request.is_json:
            custom_price = request.json.get('custom_price')
        if custom_price is None:
            custom_price = request.args.get('custom_price')
            
        df, date_col = load_dataset()
        latest_real_price = float(df['Close'].iloc[-1])
        latest_date = df[date_col].iloc[-1]

        if custom_price is None or str(custom_price).strip() == '':
            custom_price = latest_real_price
        else:
            custom_price = float(custom_price)

        # Ambil 29 hari historis sebelum hari ini + 1 hari skenario kustom (total 30 hari window)
        prev_29_days = df['Close'].tail(TIME_STEP - 1).values.tolist()
        simulated_window = prev_29_days + [custom_price]

        # Scaling input
        window_arr = np.array(simulated_window).reshape(-1, 1)
        scaled_input = scaler.transform(window_arr).astype(np.float32).reshape(1, TIME_STEP, 1)

        # Prediksi inferensi LSTM langsung (cepat & hemat memori)
        pred_scaled = model(scaled_input, training=False).numpy()
        simulated_prediction = float(scaler.inverse_transform(pred_scaled)[0][0])

        # Hitung proyeksi selisih dan persentase
        price_diff = simulated_prediction - custom_price
        pct_diff = (price_diff / custom_price) * 100
        trend = 'UP' if price_diff >= 0 else 'DOWN'

        next_date = (latest_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify({
            'status': 'success',
            'custom_price': round(custom_price, 2),
            'latest_real_price': round(latest_real_price, 2),
            'next_date': next_date,
            'simulated_prediction': round(simulated_prediction, 2),
            'price_change': round(price_diff, 2),
            'percentage_change': round(pct_diff, 2),
            'trend': trend
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Jalankan server Flask pada port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
