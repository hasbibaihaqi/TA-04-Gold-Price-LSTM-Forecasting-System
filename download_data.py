import yfinance as yf
import pandas as pd

def download_gold_data():
    # Ticker untuk Emas (Gold Futures) di Yahoo Finance adalah GC=F
    ticker = "GC=F"
    
    print(f"Memulai proses unduhan data untuk ticker: {ticker}...")
    
    # Mengunduh data historis harian untuk 3 tahun terakhir
    gold_data = yf.download(ticker, period="3y", interval="1d")
    
    # Memeriksa apakah data berhasil diambil
    if gold_data.empty:
        print("Gagal mengunduh data. Periksa koneksi internet atau simbol ticker.")
        return False, "Gagal mengunduh data dari Yahoo Finance."
    
    # Menghapus missing values (NaN)
    gold_data = gold_data.dropna()
    
    # Menangani MultiIndex columns dari yfinance versi baru
    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_data.columns = gold_data.columns.get_level_values(0)
    
    # Reset index agar 'Date' menjadi kolom biasa yang bersih
    gold_data.reset_index(inplace=True)
    
    # Format kolom Date agar formatnya konsisten string YYYY-MM-DD
    gold_data['Date'] = pd.to_datetime(gold_data['Date']).dt.strftime('%Y-%m-%d')
    
    # Nama file output
    file_name = "dataset_timeseries.csv"
    
    # Menyimpan DataFrame ke file CSV tanpa index tambahan
    gold_data.to_csv(file_name, index=False)
    
    latest_row = gold_data.iloc[-1]
    info = {
        'total_rows': len(gold_data),
        'latest_date': str(latest_row['Date']),
        'latest_price': round(float(latest_row['Close']), 2)
    }
    
    print(f"\n[SUKSES] Data berhasil diunduh dan disimpan sebagai '{file_name}'")
    print(f"Total baris data: {info['total_rows']}, Tanggal Terakhir: {info['latest_date']}")
    return True, info

if __name__ == "__main__":
    download_gold_data()
