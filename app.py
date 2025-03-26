import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
import os
import requests
import numpy as np

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

def transformar_datos(df):
    # Eliminar columna Id si existe
    df = df.drop('Id', axis=1)
    df=df.fillna(0)

    # Variables numéricas clave
    numerical_columns = df.select_dtypes(include=['number']).columns
    # Variables binarias
    binary_columns = ['CentralAir']
    # Variables que deberían ser convertidas en dummies
    dummy_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
    
    # Crear las dummy variables para las columnas categóricas
    df_dummies = pd.get_dummies(df[dummy_columns], prefix=dummy_columns)
    # Convertir columnas booleanas (True/False) a 1/0 usando apply
    df_dummies = df_dummies.apply(lambda col: col.map({True: 1, False: 0}) if col.dtype == bool else col)
    # Unir las nuevas columnas dummy al DataFrame original
    df = pd.concat([df, df_dummies], axis=1)
    # Eliminar las columnas originales de tipo categórico si ya no las necesitas
    df = df.drop(columns=dummy_columns)

    # Convertir variables categóricas binarias (Y/N) a 0 y 1
    df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'Y': 1, 'N': 0}))
    df[numerical_columns] = df[numerical_columns].astype(int)

    # Crear características sumatorias
    df['TotalSF'] = df['GrLivArea']+df['GarageArea']+df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']+df['BsmtFinSF1'] #superficie total de la vivienda
    df['TotalBaths']=df['FullBath']+(df['HalfBath']*0.5)+(df['BsmtHalfBath']*0.5)+df['BsmtFullBath'] #cantidad de baños totales
    df['TotalPorchSF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] #superficie total de porches
    
    # Crear características derivadas
    df['HouseAge']=df['YrSold']-df['YearBuilt'] #edad casa
    df['TimeSinceRemod']=df['YrSold']-df['YearRemodAdd'] #tiempo desde remodelacion
    df['GarageAreaPerCar']=df['GarageArea']/(df['GarageCars']+1) #ratio garaje/coche
    df['LotDensity']=df['LotArea']/df['GrLivArea'] #terreno/vivienda
    df['BsmtRatio']=(df['BsmtFinSF1']+df['BsmtFinSF2'])/df['TotalBsmtSF'] #ratio sotanos terminados
    df['QualityRatio']=df['OverallQual']*(df['ExterQual_Ex']+df['BsmtQual_Ex']+df['KitchenQual_Ex']) #ratio calidad general * ratio calidades excelentes
    df['SinceRemodel']=2025-df['YearRemodAdd'] #años desde remodelación
    df['AreaQualRatio']=df['GrLivArea']/df['OverallQual'] #ratio espacio/calidad
    df['LotToAreaRatio']=df['LotArea']/df['GrLivArea'] #ratio espacio/espacio habitable
    
    df=df.fillna(0)
    
    return df

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

    # Guardar el precio de venta para devolverlo en la respuesta
    sale_price = df['SalePrice'].copy() if 'SalePrice' in df.columns else None

    # Eliminar columna SalePrice para predecir
    X = df.drop(columns=['SalePrice'], errors='ignore')

    # Aplicar transformaciones
    X_transformado = transformar_datos(X)

    # Realizar predicciones
    predictions = model.predict(X_transformado)

    # Añadir predicciones al DataFrame original
    # Añadir predicciones al DataFrame original
    if sale_price is not None:
        df['SalePrice'] = sale_price  # Restaurar la columna original
    df['PredictedSalePrice'] = predictions


    # Devolver las predicciones como JSON
    return jsonify({
        'predictions': predictions.tolist(), 
        'original_data': df.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)