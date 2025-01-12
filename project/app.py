from flask import Flask, request, render_template, send_file
import pandas as pd
import re
import joblib
import os
from nltk.corpus import stopwords
import nltk

# Inisialisasi Flask
app = Flask(__name__)

# Pastikan direktori data tersedia
os.makedirs('data', exist_ok=True)

# Download stopwords jika belum diunduh
nltk.download('stopwords')

# Fungsi preprocessing teks
def preprocess(text, stopwords_list):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Hanya huruf
    text = text.lower()  # Huruf kecil
    text = " ".join([word for word in text.split() if word not in stopwords_list])  # Hilangkan stopwords
    return text

# Route: Halaman utama (Pre-Processing)
@app.route('/')
def index():
    return render_template('index.html')

# Route: Proses Pre-Processing dan Analisis Sentimen
@app.route('/process', methods=['POST'])
def process():
    # Ambil file stopword (opsional)
    stopword_file = request.files.get('stopwordFile')
    stopwords_list = set(stopwords.words('english'))
    if stopword_file:
        stopword_content = stopword_file.read().decode('utf-8').splitlines()
        stopwords_list.update(stopword_content)

    # Ambil file data
    data_file = request.files.get('dataFile')
    if not data_file:
        return "No data file uploaded."

    # Baca file data
    data = data_file.read().decode('utf-8').splitlines()
    df = pd.DataFrame(data, columns=['Kalimat'])

    # Preprocessing
    df['Cleaned'] = df['Kalimat'].apply(lambda x: preprocess(x, stopwords_list))

    # Analisis Sentimen
    model = joblib.load('models/model.pkl')  # Pastikan model tersedia
    vectorizer = joblib.load('models/vectorizer.pkl')
    X = vectorizer.transform(df['Cleaned'])
    df['Sentimen'] = model.predict(X)
    df['Sentimen'] = df['Sentimen'].map({1: 'Positif', 0: 'Negatif'})

    # Simpan hasil pre-processing dan analisis
    df.to_csv('data/processed_data.csv', index=False)

    # Tampilkan hasil di halaman
    return render_template('results.html', tables=df.to_dict(orient='records'))

# Route: Hasil Data Tables
@app.route('/results')
def results():
    # Baca data hasil pre-processing
    processed_data = pd.read_csv('data/processed_data.csv')

    # Kirim data ke template
    return render_template('results.html', tables=processed_data.to_dict(orient='records'))
@app.route('/visualization')
def visualization():
    try:
        # Baca file hasil analisis sentimen
        df = pd.read_csv('data/processed_data.csv')

        # Hitung jumlah setiap sentimen
        sentiment_counts = df['Sentimen'].value_counts()
        total = sentiment_counts.sum()

        # Hitung persentase untuk setiap sentimen
        percentages = {
            "Positif": round((sentiment_counts.get("Positif", 0) / total) * 100, 2),
            "Negatif": round((sentiment_counts.get("Negatif", 0) / total) * 100, 2),
            "Netral": round((sentiment_counts.get("Netral", 0) / total) * 100, 2) if "Netral" in sentiment_counts else 0
        }

        # Kirim data ke template visualization.html
        return render_template(
            'visualization.html',
            percentages=percentages,
            sentiment_counts=sentiment_counts.to_dict(),
            total=total
        )
    except Exception as e:
        return f"Error processing data for visualization: {str(e)}"

# Route: Download Hasil Lengkap
@app.route('/download')
def download():
    # File hasil analisis lengkap
    file_path = 'data/processed_data.csv'
    return send_file(file_path, as_attachment=True, download_name='hasil_analisis.csv')

if __name__ == '__main__':
    app.run(debug=True)
