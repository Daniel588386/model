import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
import os
import requests
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# URL y ruta del modelo
MODEL_URL = 'https://github.com/Daniel588386/model/raw/refs/heads/main/model.keras'
MODEL_PATH = 'model.keras'

def download_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Modelo descargado en {MODEL_PATH}")
    else:
        print("Error al descargar el modelo")

# Descargar el modelo solo la primera vez
if not os.path.exists(MODEL_PATH):
    download_model()

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Función para normalizar los datos con StandardScaler
def scale_data(X):
    # Eliminar columnas no numéricas
    X_numeric = X.select_dtypes(include=[np.number])

    # Manejar valores nulos (NaN) rellenándolos con la mediana de cada columna
    X_numeric = X_numeric.fillna(X_numeric.median())

    # Normalización con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    return X_scaled

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Leer el archivo CSV usando pandas
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error al leer el archivo CSV: {str(e)}'}), 400

    # Preprocesar los datos
    X = df.drop(columns=['SalePrice'])  # Asegúrate de ajustar la columna de acuerdo con tus datos
    X_scaled = scale_data(X)  # Aplica el preprocesamiento simplificado

    # Verificar los datos escalados
    print("Datos escalados: ", X_scaled[:5])  # Muestra las primeras filas de los datos escalados para asegurarte de que no sean iguales

    # Realizar predicciones
    predictions = model.predict(X_scaled).flatten()

    # Devolver las predicciones como JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
