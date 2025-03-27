# 1 Business Context
Eres parte de un equipo de análisis en una firma de inversión inmobiliaria. La empresa busca mejorar la precisión en la valoración de propiedades con el objetivo de maximizar la rentabilidad y minimizar riesgos. En este mercado, errores en la estimación de precios pueden resultar en pérdidas millonarias.

La empresa dispone de un dataset histórico (train):

https://www.kaggle.com/competitions/home-data-for-ml-course/data

## 1.2 Objective
Develop a neural network model to predict housing prices with greater accuracy than traditional methods such as linear regression.

# 2 Data Processing
In [Predict.ipynb](https://github.com/Daniel588386/model/blob/main/Predict.ipynb) we got the whole process. 
Out dataset contains:
* **Características de entrada:**
* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class
* MSZoning: The general zoning classification
* LotFrontage: Linear feet of street connected to property
* LotArea: Lot size in square feet
* Street: Type of road access
* Alley: Type of alley access
* LotShape: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
* RoofMatl: Roof material
* Exterior1st: Exterior covering on house
* Exterior2nd: Exterior covering on house (if more than one material)
* MasVnrType: Masonry veneer type
* MasVnrArea: Masonry veneer area in square feet
* ExterQual: Exterior material quality
* ExterCond: Present condition of the material on the exterior
* Foundation: Type of foundation
* BsmtQual: Height of the basement
* BsmtCond: General condition of the basement
* BsmtExposure: Walkout or garden level basement walls
* BsmtFinType1: Quality of basement finished area
* BsmtFinSF1: Type 1 finished square feet
* BsmtFinType2: Quality of second finished area (if present)
* BsmtFinSF2: Type 2 finished square feet
* BsmtUnfSF: Unfinished square feet of basement area
* TotalBsmtSF: Total square feet of basement area
* Heating: Type of heating
* HeatingQC: Heating quality and condition
* CentralAir: Central air conditioning
* Electrical: Electrical system
* 1stFlrSF: First Floor square feet
* 2ndFlrSF: Second floor square feet
* LowQualFinSF: Low quality finished square feet (all floors)
* GrLivArea: Above grade (ground) living area square feet
* BsmtFullBath: Basement full bathrooms
* BsmtHalfBath: Basement half bathrooms
* FullBath: Full bathrooms above grade
* HalfBath: Half baths above grade
* Bedroom: Number of bedrooms above basement level
* Kitchen: Number of kitchens
* KitchenQual: Kitchen quality
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
* Functional: Home functionality rating
* Fireplaces: Number of fireplaces
* FireplaceQu: Fireplace quality
* GarageType: Garage location
* GarageYrBlt: Year garage was built
* GarageFinish: Interior finish of the garage
* GarageCars: Size of garage in car capacity
* GarageArea: Size of garage in square feet
* GarageQual: Garage quality
* GarageCond: Garage condition
* PavedDrive: Paved driveway
* WoodDeckSF: Wood deck area in square feet
* OpenPorchSF: Open porch area in square feet
* EnclosedPorch: Enclosed porch area in square feet
* 3SsnPorch: Three season porch area in square feet
* ScreenPorch: Screen porch area in square feet
* PoolArea: Pool area in square feet
* PoolQC: Pool quality
* Fence: Fence quality
* MiscFeature: Miscellaneous feature not covered in other categories
* MiscVal: $Value of miscellaneous feature
* MoSold: Month Sold
* YrSold: Year Sold
* SaleType: Type of sale
* SaleCondition: Condition of sale


Data transformations:
````python
# Eliminate Id, for it has no use for prediction purposes
df = df.drop('Id', axis=1)

print("INFO DEL DATASET:")
df.info()
````
Create dummy variables for categorical columns:
````python
# Key categorical
numerical_columns = df.select_dtypes(include=['number']).columns

# Binary
binary_columns = ['CentralAir']

# Dummy variables candidates
dummy_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Create dummy variables
df_dummies = pd.get_dummies(df[dummy_columns], prefix=dummy_columns)

# True/False to 1/0
df_dummies = df_dummies.apply(lambda col: col.map({True: 1, False: 0}) if col.dtype == bool else col)

df = pd.concat([df, df_dummies], axis=1)

# Drop redundant
df = df.drop(columns=dummy_columns)
````
Feature Engineering:
````python
# Summatory variables
df['TotalSF'] = df['GrLivArea']+df['GarageArea']+df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']+df['BsmtFinSF1'] #Total area
df['TotalBaths']=df['FullBath']+(df['HalfBath']*0.5)+(df['BsmtHalfBath']*0.5)+df['BsmtFullBath'] #total bathrooms
df['TotalPorchSF'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] #total porch area

# Derived features
df['HouseAge']=df['YrSold']-df['YearBuilt'] #age
df['TimeSinceRemod']=df['YrSold']-df['YearRemodAdd'] #since reomd
df['GarageAreaPerCar']=df['GarageArea']/(df['GarageCars']+1) #car/garage ratio
df['LotDensity']=df['LotArea']/df['GrLivArea'] #lot/living area ratio
df['BsmtRatio']=(df['BsmtFinSF1']+df['BsmtFinSF2'])/df['TotalBsmtSF'] #finished basement ratio
df['QualityRatio']=df['OverallQual']*(df['ExterQual_Ex']+df['BsmtQual_Ex']+df['KitchenQual_Ex']) #quality/ex quality ratio
df['SinceRemodel']=2025-df['YearRemodAdd'] #since actual remod
df['AreaQualRatio']=df['GrLivArea']/df['OverallQual'] #space/quad ratio
df['LotToAreaRatio']=df['LotArea']/df['GrLivArea'] #total space/living space ratio
````
This feature engineering brings us some heavy-weight correlation values:

* TotalSF - 0.801297
* OverallQual - 0.790982
* GrLivArea - 0.708624
* GarageCars - 0.640409
* TotalBaths - 0.631731
* GarageArea - 0.623431
* QualityRatio - 0.620616
* TotalBsmtSF - 0.613581
* 1stFlrSF - 0.605852
* FullBath - 0.560664
* BsmtQual_Ex - 0.553105
* TotRmsAbvGrd - 0.533723
* YearBuilt - 0.522897
* YearRemodAdd - 0.507101
* KitchenQual_Ex - 0.504094

Remove outliers (applying 10%)
````python
def remove_outliers_iqr(df, columns):
    # Identify binary columns (0/1)
    binary_columns = [col for col in columns if df[col].nunique() == 2]

    # Filtering non-binary columns
    numerical_columns = [col for col in columns if col not in binary_columns]

    if numerical_columns:
        Q1 = df[numerical_columns].quantile(0.10)
        Q3 = df[numerical_columns].quantile(0.90)
        IQR = Q3 - Q1

        # Define outlier limits
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filtering in range
        df = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]

    return df

# Exclude binary columns
df = remove_outliers_iqr(df, numerical_columns)
````
Define training/test data size:
````python
from sklearn.model_selection import train_test_split

# 80%traint 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display Ammount of data in each
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")
````
The model is trained with these parameters:
* K-Fold: 5
* Learning rate: 0.001
* Dropout: 0.2
* L2: 0.01
* Batch size: 100
* Activation: relu
* Model structure: 3 dense layers (512, 256, 256). 1 dense exit
* Epoch: 400 (with early stopping)

These parameters have been tested among many other combinations (previous cell in notebook), choosing the best result according to the case study.

Final model result:
````
Final result:
MSE: 12.4275 %
RMSE: 12.3662 %
R²: 87.5725 %
Avg Epoch: 138.0
````

# 3 API and Client use
[app.py](https://github.com/Daniel588386/model/blob/main/app.py), [client.py](https://github.com/Daniel588386/model/blob/main/client.py) and [train.csv](https://github.com/Daniel588386/model/blob/main/train.csv) should be in the same directory.
* In order to set the API running just run the app.py code.
* In order to predict data run client.py and select in the console the .csv file detected in the folder
* Once selected it whould show predicted SalePrice vs real SalePrice, like this:
````
--- Predicted vs. Actual Prices ---
      SalePrice  PredictedSalePrice
0        208500       210596.656250
1        181500       165269.593750
2        223500       218611.656250
3        140000       162525.218750
4        250000       290874.156250
````
