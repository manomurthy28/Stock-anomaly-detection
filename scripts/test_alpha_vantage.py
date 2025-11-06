import requests
import pandas as pd
import time
from datetime import datetime
import os
import json

# Use the provided API key
API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'L0WU5V9YC0JTXS4Y')
BASE_URL = 'https://www.alphavantage.co/query'

def fetch_stock_data(stock_symbol, output_size='full'):
    """
    Fetches stock data from Alpha Vantage API for a given stock symbol.
    
    Parameters:
    stock_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA').
    output_size (str): 'compact' for recent data (100 entries), 'full' for complete historical data.
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    print(f"Fetching data for {stock_symbol} from Alpha Vantage...")
    
    # Define API parameters
    params = {
        'function': 'TIME_SERIES_DAILY',  # Fetch daily stock data
        'symbol': stock_symbol,
        'outputsize': output_size,  # Fetch full or compact data
        'apikey': API_KEY  # API key for authentication
    }

    # Make the API request
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

    # Check if the request was successful
    if response.status_code == 200:
        try:
            data = response.json()  # Convert response to JSON format
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None

        # Check if API returned an error message
        if "Error Message" in data:
            print("API Error:", data["Error Message"])  # Display API error message
            return None
        
        # Check if API returned an information message (usually for demo keys)
        if "Information" in data:
            print("API Information:", data["Information"])
            return None

        # Extract the daily time series data
        time_series = data.get("Time Series (Daily)", {})

        # Check if data exists
        if not time_series:
            print("No time series data found!")
            return None

        try:
            # Convert the data into a pandas DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')

            # Reset the index and rename columns for readability
            df.reset_index(inplace=True)
            df.columns = ['timestamp'] + [col.split('. ')[1] for col in df.columns[1:]]

            # Convert data types for proper numerical analysis
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Sort by date (newest first)
            df.sort_index(ascending=False, inplace=True)
            
            return df
        except Exception as e:
            print(f"Error processing Alpha Vantage data: {e}")
            return None
    else:
        # Print HTTP error code if request fails
        print(f"HTTP Error: {response.status_code}")
        return None

def test_alpha_vantage():
    """
    Test the Alpha Vantage API with a few stock symbols
    """
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        data = fetch_stock_data(symbol, 'compact')
        
        if data is not None:
            print(f"Successfully fetched data for {symbol}")
            print(f"Data shape: {data.shape}")
            print("First 5 rows:")
            print(data.head())
        else:
            print(f"Failed to fetch data for {symbol}")
        
        # Wait to avoid hitting API rate limits
        time.sleep(15)

if __name__ == "__main__":
    test_alpha_vantage()
