import requests
import pandas as pd
import os

def predict_house_prices(csv_file):
    """
    Send CSV file to the prediction API and process the results
    
    Args:
        csv_file (str): Path to the CSV file to be processed
    
    Returns:
        DataFrame with original and predicted prices
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found")

    # Prepare file for upload
    with open(csv_file, 'rb') as file:
        files = {'file': file}
        
        # Send request to prediction API
        response = requests.post('http://localhost:5000/predict', files=files)
    
    # Check response
    if response.status_code != 200:
        raise Exception(f"API Error: {response.json().get('error', 'Unknown error')}")

    # Parse response
    result = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(result['original_data'])
    
    # Show original and predicted prices
    if 'SalePrice' in df.columns and 'PredictedSalePrice' in df.columns:
        print("\n--- Predicted vs. Actual Prices ---")
        print(df[['SalePrice', 'PredictedSalePrice']].head())

    return df

def main():
    # List CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print("Available CSV files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    # Let user choose file
    while True:
        try:
            choice = int(input("\nEnter the number of the CSV file to predict: ")) - 1
            selected_file = csv_files[choice]
            break
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
    
    try:
        # Predict and display results
        predict_house_prices(selected_file)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
