import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime
import traceback
import yfinance as yf

# Use the provided API key
API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'L0WU5V9YC0JTXS4Y')
BASE_URL = 'https://www.alphavantage.co/query'
OUTPUT_SIZE = 'full'  # 'compact' for recent data (100 entries), 'full' for complete historical data.

def fetch_with_direct_yahoo(stock_symbol, use_cache=True, cache_dir='./cache'):
  """
  Fetch stock data directly from Yahoo Finance API
  
  Parameters:
  stock_symbol (str): The stock ticker symbol
  use_cache (bool): Whether to use cached data if available
  cache_dir (str): Directory to store cached data
  
  Returns:
  pandas.DataFrame: DataFrame with stock data
  """
  try:
      # Check cache first
      if use_cache:
          if not os.path.exists(cache_dir):
              os.makedirs(cache_dir)
          
          cache_file = f"{cache_dir}/{stock_symbol}_yahoo_direct.csv"
          if os.path.exists(cache_file):
              file_modified_time = os.path.getmtime(cache_file)
              current_time = time.time()
              # If file is less than 1 day old, use cached data
              if (current_time - file_modified_time) < 86400:  # 86400 seconds = 1 day
                  print(f"Using cached Yahoo direct data for {stock_symbol}")
                  try:
                      return pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')
                  except Exception as e:
                      print(f"Error reading Yahoo direct cache: {e}")
                      # Continue to fetch new data
      
      print(f"Fetching data directly from Yahoo Finance API for {stock_symbol}...")
      
      # For Indian stocks, try different intervals
      period = "1y"
      
      # Construct Yahoo Finance API URL
      query = {
          "symbols": stock_symbol,
          "range": period,
          "interval": "1d",
          "indicators": "quote",
          "includeTimestamps": True,
          "includePrePost": False,
          "corsDomain": "finance.yahoo.com"
      }
      
      url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_symbol}"
      
      headers = {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
      }
      
      try:
          response = requests.get(url, headers=headers, params=query, timeout=30)
      except requests.exceptions.RequestException as e:
          print(f"Request error with Yahoo direct API: {e}")
          return None
      
      if response.status_code != 200:
          print(f"Direct Yahoo API returned status code {response.status_code}")
          return None
          
      try:
          data = response.json()
      except json.JSONDecodeError as e:
          print(f"JSON decode error with Yahoo direct API: {e}")
          return None
      
      # Check if we have valid data
      if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
          print(f"No data returned from Yahoo Finance API for {stock_symbol}")
          return None
          
      result = data["chart"]["result"][0]
      timestamps = result.get("timestamp", [])
      
      if not timestamps:
          print(f"No timestamps found in response for {stock_symbol}")
          return None
          
      quote = result.get("indicators", {}).get("quote", [{}])[0]
      
      # Check if we have price data
      if not quote.get("close"):
          print(f"No price data found in response for {stock_symbol}")
          return None
          
      # Convert to DataFrame
      try:
          df = pd.DataFrame({
              'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
              'open': quote.get('open', []),
              'high': quote.get('high', []),
              'low': quote.get('low', []),
              'close': quote.get('close', []),
              'volume': quote.get('volume', [])
          })
          
          # Handle missing values
          df = df.dropna()
          
          # Set timestamp as index
          df.set_index('timestamp', inplace=True)
          
          # Sort by date (newest first)
          df.sort_index(ascending=False, inplace=True)
          
          print(f"Successfully fetched data directly from Yahoo Finance API for {stock_symbol}")
          print(f"Data shape: {df.shape}")
          print(f"Date range: {df.index.min()} to {df.index.max()}")
          
          # Save to cache if enabled
          if use_cache:
              df.to_csv(cache_file)
              print(f"Data saved to cache: {cache_file}")
          
          return df
      except Exception as e:
          print(f"Error processing Yahoo direct data: {e}")
          traceback.print_exc()
          return None
          
  except Exception as e:
      print(f"Error fetching data directly from Yahoo Finance API: {e}")
      traceback.print_exc()
      return None

def fetch_with_yfinance(stock_symbol, use_cache=True, cache_dir='./cache'):
  """
  Fetch stock data using yfinance as a fallback
  
  Parameters:
  stock_symbol (str): The stock ticker symbol
  use_cache (bool): Whether to use cached data if available
  cache_dir (str): Directory to store cached data
  
  Returns:
  pandas.DataFrame: DataFrame with stock data
  """
  try:
      print(f"Fetching data for {stock_symbol} using yfinance...")
      
      # Download data
      try:
          data = yf.download(stock_symbol, period="1y", progress=False, timeout=30)
      except Exception as e:
          print(f"Error downloading with yfinance: {e}")
          return None
      
      if data.empty:
          print(f"No data found for {stock_symbol} using yfinance")
          return None
      
      # Rename columns to match our format
      data.columns = [col.lower() for col in data.columns]
      
      # Sort by date (newest first)
      data.sort_index(ascending=False, inplace=True)
      
      # Save to cache if enabled
      if use_cache:
          if not os.path.exists(cache_dir):
              os.makedirs(cache_dir)
          cache_file = f"{cache_dir}/{stock_symbol}_daily_data.csv"
          data.to_csv(cache_file)
          print(f"Data saved to cache: {cache_file}")
      
      return data
  except Exception as e:
      print(f"Error fetching data with yfinance: {e}")
      return None

def fetch_stock_data(stock_symbol, use_cache=True, cache_dir='./cache'):
  """
  Fetches stock data from Alpha Vantage API for a given stock symbol.
  
  Parameters:
  stock_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA').
  use_cache (bool): Whether to use cached data if available
  cache_dir (str): Directory to store cached data
  
  Returns:
  pandas.DataFrame: DataFrame with stock data
  """
  # Create cache directory if it doesn't exist
  if use_cache and not os.path.exists(cache_dir):
      os.makedirs(cache_dir)
  
  # Check if cached data exists and is recent (less than 1 day old)
  cache_file = f"{cache_dir}/{stock_symbol}_daily_data.csv"
  if use_cache and os.path.exists(cache_file):
      file_modified_time = os.path.getmtime(cache_file)
      current_time = time.time()
      # If file is less than 1 day old, use cached data
      if (current_time - file_modified_time) < 86400:  # 86400 seconds = 1 day
          print(f"Using cached data for {stock_symbol}")
          try:
              return pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')
          except Exception as e:
              print(f"Error reading cache file: {e}")
              # Continue to fetch new data

  print(f"Fetching data for {stock_symbol} from Alpha Vantage...")
  
  # Define API parameters
  params = {
      'function': 'TIME_SERIES_DAILY',  # Fetch daily stock data
      'symbol': stock_symbol,
      'outputsize': OUTPUT_SIZE,  # Fetch full or compact data
      'apikey': API_KEY  # API key for authentication
  }

  # Make the API request
  try:
      response = requests.get(BASE_URL, params=params, timeout=30)
  except requests.exceptions.RequestException as e:
      print(f"Request error: {e}")
      print("Trying yfinance as fallback...")
      return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)

  # Check if the request was successful
  if response.status_code == 200:
      try:
          data = response.json()  # Convert response to JSON format
      except json.JSONDecodeError as e:
          print(f"JSON decode error: {e}")
          print("Trying yfinance as fallback...")
          return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)

      # Check if API returned an error message
      if "Error Message" in data:
          print("API Error:", data["Error Message"])  # Display API error message
          print("Trying yfinance as fallback...")
          return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)
      
      # Check if API returned an information message (usually for demo keys)
      if "Information" in data:
          print("API Information:", data["Information"])
          print("Trying yfinance as fallback...")
          return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)

      # Extract the daily time series data
      time_series = data.get("Time Series (Daily)", {})

      # Check if data exists
      if not time_series:
          print("No time series data found! Trying yfinance as fallback...")
          return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)

      try:
          # Convert the data into a pandas DataFrame
          df = pd.DataFrame.from_dict(time_series, orient='index')

          # Reset the index and rename columns for readability
          df.reset_index(inplace=True)
          df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

          # Convert data types for proper numerical analysis
          df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
          df['volume'] = df['volume'].astype(int)
          
          # Convert timestamp to datetime and set as index
          df['timestamp'] = pd.to_datetime(df['timestamp'])
          df.set_index('timestamp', inplace=True)
          
          # Sort by date (newest first)
          df.sort_index(ascending=False, inplace=True)
          
          # Limit to 1 year of data to match yfinance behavior
          one_year_ago = datetime.now() - pd.DateOffset(years=1)
          df = df[df.index >= one_year_ago]
          
          # Save to cache if enabled
          if use_cache:
              df.to_csv(cache_file)
              print(f"Data saved to cache: {cache_file}")

          return df
      except Exception as e:
          print(f"Error processing Alpha Vantage data: {e}")
          print("Trying yfinance as fallback...")
          return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)
  else:
      # Print HTTP error code if request fails
      print(f"HTTP Error: {response.status_code}")
      print("Trying yfinance as fallback...")
      return fetch_with_yfinance(stock_symbol, use_cache, cache_dir)

def get_stock_data(stock_symbol):
  """
  Get stock data with retry mechanism
  
  Parameters:
  stock_symbol (str): The stock ticker symbol
  
  Returns:
  pandas.DataFrame: DataFrame with stock data
  """
  # For Indian stocks, ONLY use direct Yahoo Finance API
  if stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO'):
      print(f"Indian stock detected: {stock_symbol}. Using direct Yahoo Finance API...")
      
      # Try direct Yahoo API
      data = fetch_with_direct_yahoo(stock_symbol, use_cache=True)
      if data is not None and not data.empty:
          print(f"Successfully fetched Indian stock data for {stock_symbol} using direct Yahoo API")
          return data
      
      print("Direct Yahoo API failed. Trying again with different parameters...")
      # Try again with different parameters
      data = fetch_with_direct_yahoo(stock_symbol, use_cache=False)
      if data is not None and not data.empty:
          print(f"Successfully fetched Indian stock data for {stock_symbol} on second attempt")
          return data
          
      print("All Yahoo Finance direct API attempts failed.")
      return None
  
  # For non-Indian stocks, try Alpha Vantage first, then yfinance
  max_retries = 3
  for attempt in range(max_retries):
      try:
          # Try Alpha Vantage
          data = fetch_stock_data(stock_symbol)
          if data is not None and not data.empty:
              return data
          
          print(f"Attempt {attempt + 1} failed. Retrying...")
          time.sleep(2)  # Wait before retrying
      except Exception as e:
          print(f"Error on attempt {attempt + 1}: {e}")
          if attempt < max_retries - 1:
              print("Retrying...")
              time.sleep(2)
          else:
              print("All retry attempts failed.")
              return None
  
  return None

# Test function
if __name__ == "__main__":
  symbol = input("Enter stock symbol (e.g., AAPL or SBIN.NS): ").strip().upper()
  data = get_stock_data(symbol)
  if data is not None:
      print(f"Successfully fetched data for {symbol}")
      print(f"Data shape: {data.shape}")
      print("First 5 rows:")
      print(data.head())
  else:
      print(f"Failed to fetch data for {symbol}")
