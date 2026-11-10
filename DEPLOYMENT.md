# Panduan Deployment Aplikasi Web (GoldPredict AI)

Dokumen ini menjelaskan langkah-langkah mendeploy aplikasi peramalan LSTM ini ke platform cloud gratis agar dapat diakses publik melalui internet oleh dosen atau rekan mahasiswa.

---

## Opsi 1: Deploy ke Render.com (Gratis & Direkomendasikan)

[Render.com](https://render.com) mendukung aplikasi Python/Flask dan integrasi otomatis dengan repositori GitHub.

### Langkah-langkah:
1. **Unggah Proyek ke GitHub:**
   - Buat repositori baru di akun GitHub Anda (misal: `ta04-lstm-gold`).
   - Push seluruh file proyek ini ke repositori tersebut:
     ```bash
     git init
     git add .
     git commit -m "Initial commit TA-04 LSTM Gold Predictor"
     git branch -M main
     git remote add origin https://github.com/<username-anda>/ta04-lstm-gold.git
     git push -u origin main
     ```

2. **Buat Web Service di Render:**
   - Buka [dashboard.render.com](https://dashboard.render.com) dan login menggunakan akun GitHub.
   - Klik tombol **New +** -> pilih **Web Service**.
   - Pilih repositori `ta04-lstm-gold` yang baru diunggah.

3. **Konfigurasi Build & Run Settings:**
   - **Name:** `goldpredict-ai` (atau nama lain pilihan Anda).
   - **Environment:** `Python 3`
   - **Region:** `Singapore` (terdekat dengan Indonesia untuk latensi rendah).
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     gunicorn app:app
     ```
   - **Instance Type:** `Free`

4. **Deploy:**
   - Klik tombol **Deploy Web Service**.
   - Tunggu beberapa menit hingga proses build dan unduh dependensi selesai.
   - Aplikasi Anda akan aktif dengan alamat URL publik, misalnya:
     👉 `https://goldpredict-ai.onrender.com`

---

## Opsi 2: Deploy ke PythonAnywhere (Gratis)

[PythonAnywhere](https://www.pythonanywhere.com) menyediakan hosting Flask berbasis Linux gratis tanpa kartu kredit:
1. Daftar akun gratis di [PythonAnywhere](https://www.pythonanywhere.com).
2. Buka tab **Consoles** -> pilih **Bash Console**.
3. Clone repositori proyek Anda:
   ```bash
   git clone https://github.com/<username-anda>/ta04-lstm-gold.git
   ```
4. Buka tab **Web** -> klik **Add a new web app** -> pilih **Flask** -> **Python 3.11**.
5. Sesuaikan path working directory ke `/home/<username>/ta04-lstm-gold`.
6. Install dependensi di console:
   ```bash
   pip install -r requirements.txt
   ```
7. Klik tombol hijau **Reload** pada halaman tab Web.

---

## 🔄 Pembaruan Data Setelah Dideploy
Setelah aplikasi berhasil online di cloud:
* Anda tidak perlu lagi login ke terminal server.
* Kapan pun Anda atau dosen ingin memprediksi harga emas terbaru, cukup buka URL web tersebut dan klik tombol **`Sync Yahoo Finance`** di pojok kanan atas.
* Data bursa komoditas akan otomatis diperbarui dan model LSTM langsung menghitung peramalannya.

