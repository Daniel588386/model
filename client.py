import requests

url = 'http://127.0.0.1:5000/predict'  # La URL de tu API local

# Leer el archivo CSV
file_path = 'test.csv'  # Asegúrate de tener este archivo en el directorio correcto
with open(file_path, 'rb') as f:
    # Enviar una solicitud POST a la API con el archivo CSV
    response = requests.post(url, files={'file': f})

# Verificar la respuesta de la API
if response.status_code == 200:
    result = response.json()
    print("Predicciones del modelo:")
    print(result['predictions'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
import requests

url = 'http://127.0.0.1:5000/predict'  # La URL de tu API local

# Leer el archivo CSV
file_path = 'test.csv'  # Asegúrate de tener este archivo en el directorio correcto
with open(file_path, 'rb') as f:
    # Enviar una solicitud POST a la API con el archivo CSV
    response = requests.post(url, files={'file': f})

# Verificar la respuesta de la API
if response.status_code == 200:
    result = response.json()
    print("Predicciones del modelo:")
    print(result['predictions'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
