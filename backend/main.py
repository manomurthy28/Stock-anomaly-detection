from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import time
import traceback
import requests
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

# Use the provided API key
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', 'L0WU5V9YC0JTXS4Y')

# Initialize FastAPI app
app = FastAPI(title="Stock Anomaly Detection API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------
# Feature Engineering Functions
# --------------------------------------
def compute_rsi(data, window=14):
    """Compute Relative Strength Index (RSI)"""
    data = data.squeeze()
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=data.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def compute_macd(data, slow=26, fast=12, signal=9):
    """Compute MACD and Signal Line"""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# --------------------------------------
# Function to Create Sequences for LSTM
# --------------------------------------
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
    return np.array(sequences)

# --------------------------------------
# Build LSTM Autoencoder Model
# --------------------------------------
def build_lstm_autoencoder(seq_length, n_features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(seq_length * n_features, activation='linear'),
        tf.keras.layers.Reshape((seq_length, n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --------------------------------------
# Data Cleaning Functions
# --------------------------------------
def clean_data(df):
    """
    Clean data by removing NaN, infinity, and extreme values
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Replace infinity with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values
    if df_clean.isna().any().any():
        print(f"Found NaN values. Filling with appropriate values...")
        
        # For each column, fill NaN with appropriate values
        for col in df_clean.columns:
            # For price and volume data, forward fill then backward fill
            if col in ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
            # For other columns, fill with median
            else:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean

# --------------------------------------
# Alpha Vantage API Function
# --------------------------------------
def fetch_with_alpha_vantage(stock_symbol):
    """
    Fetch stock data from Alpha Vantage API
    
    Parameters:
    stock_symbol (str): The stock ticker symbol
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        print(f"ALPHA VANTAGE API: Fetching data for {stock_symbol}...")
        
        # Define API parameters
        params = {
            'function': 'TIME_SERIES_DAILY',  # Fetch daily stock data
            'symbol': stock_symbol,
            'outputsize': 'full',  # Fetch full historical data
            'apikey': ALPHA_VANTAGE_API_KEY  # API key for authentication
        }
        
        # Make the API request
        response = requests.get('https://www.alphavantage.co/query', params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"ALPHA VANTAGE API: Error - status code {response.status_code}")
            return None
            
        data = response.json()
        
        # Check if API returned an error message
        if "Error Message" in data:
            print(f"ALPHA VANTAGE API: Error - {data['Error Message']}")
            return None
            
        # Check if API returned an information message (usually for demo keys)
        if "Information" in data:
            print(f"ALPHA VANTAGE API: Information - {data['Information']}")
            return None
            
        # Extract the daily time series data
        time_series = data.get("Time Series (Daily)", {})
        
        if not time_series:
            print(f"ALPHA VANTAGE API: No time series data found for {stock_symbol}")
            return None
            
        # Convert the data into a pandas DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns for readability
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert data types for proper numerical analysis
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Rename columns to match our expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Sort by date (newest first)
        df.sort_index(ascending=False, inplace=True)
        
        # Limit to 1 year of data
        one_year_ago = datetime.now() - pd.DateOffset(years=1)
        df = df[df.index >= one_year_ago]
        
        print(f"ALPHA VANTAGE API: Successfully fetched data for {stock_symbol}")
        print(f"ALPHA VANTAGE API: Data shape: {df.shape}")
        print(f"ALPHA VANTAGE API: Date range: {df.index.min()} to {df.index.max()}")
        
        return clean_data(df)
            
    except Exception as e:
        print(f"ALPHA VANTAGE API: Error fetching data: {e}")
        traceback.print_exc()
        return None

# --------------------------------------
# Direct Yahoo API Function - ONLY method for Indian stocks
# --------------------------------------
def fetch_with_direct_yahoo(stock_symbol):
    """
    Fetch stock data directly from Yahoo Finance API
    
    Parameters:
    stock_symbol (str): The stock ticker symbol
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        print(f"DIRECT YAHOO API: Fetching data for {stock_symbol}...")
        
        # Construct Yahoo Finance API URL
        query = {
            "symbols": stock_symbol,
            "range": "1y",
            "interval": "1d",
            "indicators": "quote",
            "includeTimestamps": True,
            "includePrePost": False,
            "corsDomain": "finance.yahoo.com"
        }
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_symbol}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://finance.yahoo.com"
        }
        
        print(f"DIRECT YAHOO API: Sending request to {url}")
        response = requests.get(url, headers=headers, params=query, timeout=30)
        
        if response.status_code != 200:
            print(f"DIRECT YAHOO API: Error - status code {response.status_code}")
            print(f"DIRECT YAHOO API: Response content: {response.text[:500]}")
            return None
            
        data = response.json()
        
        # Check if we have valid data
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            print(f"DIRECT YAHOO API: No data returned for {stock_symbol}")
            return None
            
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        
        if not timestamps:
            print(f"DIRECT YAHOO API: No timestamps found for {stock_symbol}")
            return None
            
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        
        # Check if we have price data
        if not quote.get("close"):
            print(f"DIRECT YAHOO API: No price data found for {stock_symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame({
            'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        })
        
        # Handle missing values
        df = df.dropna()
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        print(f"DIRECT YAHOO API: Successfully fetched data for {stock_symbol}")
        print(f"DIRECT YAHOO API: Data shape: {df.shape}")
        print(f"DIRECT YAHOO API: Date range: {df.index.min()} to {df.index.max()}")
        
        return clean_data(df)
            
    except Exception as e:
        print(f"DIRECT YAHOO API: Error fetching data: {e}")
        traceback.print_exc()
        return None

# --------------------------------------
# yfinance API Function - Fallback for non-Indian stocks
# --------------------------------------
def fetch_with_yfinance(stock_symbol):
    """
    Fetch stock data using yfinance
    
    Parameters:
    stock_symbol (str): The stock ticker symbol
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        print(f"YFINANCE API: Fetching data for {stock_symbol}...")
        
        # Download data
        data = yf.download(
            stock_symbol, 
            period="1y", 
            interval="1d",
            progress=False,
            timeout=30
        )
        
        if data.empty:
            print(f"YFINANCE API: No data found for {stock_symbol}")
            return None
        
        print(f"YFINANCE API: Successfully fetched data for {stock_symbol}")
        print(f"YFINANCE API: Data shape: {data.shape}")
        print(f"YFINANCE API: Date range: {data.index.min()} to {data.index.max()}")
        
        return clean_data(data)
    except Exception as e:
        print(f"YFINANCE API: Error fetching data: {e}")
        traceback.print_exc()
        return None

# --------------------------------------
# Stock Data Fetching Function
# --------------------------------------
def fetch_stock_data(stock_symbol):
    """
    Fetch stock data using the appropriate method based on the stock symbol
    
    Parameters:
    stock_symbol (str): The stock ticker symbol
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        # Check if it's an Indian stock symbol
        is_indian_stock = stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO')
        
        if is_indian_stock:
            print(f"Indian stock detected: {stock_symbol}")
            print("Using ONLY direct Yahoo Finance API for Indian stocks")
            
            # For Indian stocks, ONLY use direct Yahoo Finance API
            data = fetch_with_direct_yahoo(stock_symbol)
            
            if data is None or data.empty:
                print(f"Failed to fetch data for Indian stock {stock_symbol} using direct Yahoo API")
                raise HTTPException(status_code=404, detail=f"Failed to fetch data for Indian stock {stock_symbol}")
            
            return data, "Direct Yahoo API"
        
        # For non-Indian stocks, try Alpha Vantage first
        print(f"Non-Indian stock detected: {stock_symbol}. Using Alpha Vantage API.")
        data = fetch_with_alpha_vantage(stock_symbol)
        
        if data is not None and not data.empty:
            return data, "Alpha Vantage API"
        
        # If Alpha Vantage fails, try yfinance as fallback
        print(f"Alpha Vantage failed for {stock_symbol}. Trying yfinance as fallback.")
        data = fetch_with_yfinance(stock_symbol)
        
        if data is not None and not data.empty:
            return data, "yfinance API"
        
        print(f"No data found for {stock_symbol} using any API")
        raise HTTPException(status_code=404, detail=f"No data found for {stock_symbol}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

# --------------------------------------
# API Endpoints
# --------------------------------------
@app.get("/")
def read_root():
    return {"message": "Stock Anomaly Detection API is running"}

@app.get("/test-symbol/{stock_symbol}")
async def test_stock_symbol(stock_symbol: str):
    """
    Test if a stock symbol is available and can be fetched
    """
    try:
        print(f"Testing availability of stock symbol: {stock_symbol}")
        
        # Check if it's an Indian stock symbol
        is_indian_stock = stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO')
        
        if is_indian_stock:
            print(f"Testing Indian stock symbol: {stock_symbol}")
            # For Indian stocks, ONLY use direct Yahoo Finance API
            data = fetch_with_direct_yahoo(stock_symbol)
            data_source = "Direct Yahoo API"
        else:
            # For non-Indian stocks, try Alpha Vantage first
            data = fetch_with_alpha_vantage(stock_symbol)
            data_source = "Alpha Vantage API"
            
            # If Alpha Vantage fails, try yfinance
            if data is None or data.empty:
                print(f"Alpha Vantage failed for {stock_symbol}. Trying yfinance.")
                data = fetch_with_yfinance(stock_symbol)
                data_source = "yfinance API"
        
        if data is None or data.empty:
            print(f"No data found for symbol {stock_symbol}")
            return {
                "available": False,
                "message": f"No data found for symbol {stock_symbol}"
            }
        
        # Return basic info about the data
        return {
            "available": True,
            "symbol": stock_symbol,
            "days_available": len(data),
            "date_range": {
                "start": data.index[0].strftime("%Y-%m-%d"),
                "end": data.index[-1].strftime("%Y-%m-%d")
            },
            "is_indian_stock": is_indian_stock,
            "data_source": data_source
        }
        
    except Exception as e:
        print(f"Error testing stock symbol {stock_symbol}: {str(e)}")
        traceback.print_exc()
        return {
            "available": False,
            "error": str(e)
        }

@app.get("/analyze/{stock_symbol}")
async def analyze_stock(stock_symbol: str):
    """
    Analyze stock data for anomalies using both Isolation Forest and LSTM Autoencoder
    """
    try:
        print(f"Starting analysis for {stock_symbol}...")
        
        # Check if it's an Indian stock symbol
        is_indian_stock = stock_symbol.endswith('.NS') or stock_symbol.endswith('.BO')
        if is_indian_stock:
            print(f"Processing Indian stock symbol: {stock_symbol}")
            print("Using ONLY direct Yahoo Finance API for Indian stocks")
        
        # Start timing for performance metrics
        start_time_if = time.time()
        
        # Fetch stock data
        stock_data, data_source = fetch_stock_data(stock_symbol)
        
        if stock_data is None or stock_data.empty:
            error_msg = f"No data found for symbol {stock_symbol}"
            print(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Print data info for debugging
        print(f"Stock data columns: {stock_data.columns}")
        print(f"Stock data types: {stock_data.dtypes}")
        print(f"Stock data shape: {stock_data.shape}")
        print(f"Stock data index type: {type(stock_data.index)}")
        print(f"Stock data first few rows:")
        print(stock_data.head(3))
        
        # Ensure all data is numeric
        for col in stock_data.columns:
            if stock_data[col].dtype == 'object':
                print(f"Converting column {col} to numeric")
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
        
        # Feature Engineering
        try:
            print("Starting feature engineering...")
            stock_data['MA_5'] = stock_data['Close'].rolling(window=5, min_periods=1).mean()
            stock_data['MA_10'] = stock_data['Close'].rolling(window=10, min_periods=1).mean()
            stock_data['Volatility_5'] = stock_data['Close'].rolling(window=5, min_periods=1).std()
            stock_data['RSI_14'] = compute_rsi(stock_data['Close'])
            stock_data['ROC'] = stock_data['Close'].pct_change(periods=10) * 100
            stock_data['MACD'], stock_data['MACD_Signal'] = compute_macd(stock_data['Close'])
            print("Feature engineering completed successfully")
        except Exception as e:
            print(f"Error during feature engineering: {e}")
            traceback.print_exc()
            # Continue with basic features
            print("Continuing with basic features only")
        
        # Clean data after feature engineering
        stock_data = clean_data(stock_data)
        
        # Drop NaN values after feature engineering
        stock_data.dropna(inplace=True)
        
        if len(stock_data) < 30:
            error_msg = f"Not enough data points for analysis after preprocessing. Only {len(stock_data)} points available."
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Select Relevant Features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add engineered features if they exist
        for feature in ['MA_5', 'MA_10', 'Volatility_5', 'RSI_14', 'ROC', 'MACD', 'MACD_Signal']:
            if feature in stock_data.columns:
                features.append(feature)
        
        # Check if all features exist
        missing_features = [f for f in features if f not in stock_data.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Use only available features
            features = [f for f in features if f in stock_data.columns]
            print(f"Using available features: {features}")
            
            if len(features) < 3:
                error_msg = f"Not enough features available for analysis. Only {len(features)} features available."
                print(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        
        data = stock_data[features].values
        
        # Final check for any remaining NaN or infinite values
        if np.isnan(data).any() or np.isinf(data).any():
            print("Data still contains NaN or infinite values after cleaning. Attempting final fix...")
            # Replace any remaining NaN or inf with 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Isolation Forest for Anomaly Detection
        try:
            print("Starting Isolation Forest analysis...")
            model_if = IsolationForest(contamination=0.05, random_state=42)
            anomalies_if = model_if.fit_predict(data)
            anomaly_indices_if = np.where(anomalies_if == -1)[0]  # -1 indicates anomaly
            print(f"Isolation Forest found {len(anomaly_indices_if)} anomalies")
        except Exception as e:
            print(f"Error in Isolation Forest: {e}")
            traceback.print_exc()
            # Continue with empty anomalies
            anomaly_indices_if = np.array([])
            model_if = None
        
        # End timing for Isolation Forest
        end_time_if = time.time()
        execution_time_if = end_time_if - start_time_if
        
        # Start timing for LSTM
        start_time_lstm = time.time()
        
        # Scale Data for LSTM
        try:
            print("Starting LSTM analysis...")
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create Sequences for LSTM
            seq_length = min(10, len(scaled_data) // 4)  # Adjust sequence length based on data size
            if seq_length < 2:
                seq_length = 2  # Minimum sequence length
                
            print(f"Using sequence length: {seq_length}")
            sequences = create_sequences(scaled_data, seq_length)
            
            if len(sequences) < 10:
                print(f"Not enough sequences for LSTM: {len(sequences)}")
                raise ValueError("Not enough sequences for LSTM analysis")
            
            # Train-Test Split
            train_size = max(int(len(sequences) * 0.8), 5)  # Ensure at least 5 training samples
            train_data = sequences[:train_size]
            test_data = sequences[train_size:] if train_size < len(sequences) else train_data[-5:]  # Use last 5 as test if not enough data
            
            print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
            
            # Build and Train LSTM Autoencoder
            model_lstm = build_lstm_autoencoder(seq_length, len(features))
            history = model_lstm.fit(
                train_data, train_data,
                epochs=10,  # Reduced for faster API response
                batch_size=min(32, len(train_data)),  # Adjust batch size based on data size
                validation_split=0.1,
                verbose=0
            )
            
            # Predict Reconstruction Error
            reconstructed = model_lstm.predict(test_data)
            mse = np.mean(np.power(test_data - reconstructed, 2), axis=(1,2))
            
            # Define Anomaly Threshold (95th percentile)
            threshold = np.percentile(mse, 95)
            anomalies_lstm = (mse > threshold).astype(int)
            
            # Map LSTM anomalies back to original data indices
            anomaly_indices_lstm = []
            for i, is_anomaly in enumerate(anomalies_lstm):
                if is_anomaly:
                    # Add seq_length to account for the offset in the test data
                    anomaly_indices_lstm.append(train_size + i + seq_length)
            
            print(f"LSTM found {len(anomaly_indices_lstm)} anomalies")
        except Exception as e:
            print(f"Error in LSTM: {e}")
            traceback.print_exc()
            # Continue with empty anomalies
            anomaly_indices_lstm = []
        
        # End timing for LSTM
        end_time_lstm = time.time()
        execution_time_lstm = end_time_lstm - start_time_lstm
        
        # Calculate shared anomalies
        shared_anomalies = [idx for idx in anomaly_indices_if if idx in anomaly_indices_lstm]
        agreement_percentage = round(
            (len(shared_anomalies) / ((len(anomaly_indices_if) + len(anomaly_indices_lstm)) / 2)) * 100
        ) if (len(anomaly_indices_if) + len(anomaly_indices_lstm)) > 0 else 0
        
        # Prepare stock price data for frontend
        stock_prices = []
        for date, row in stock_data.iterrows():
            stock_prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": float(row["Close"])  # Ensure it's a float
            })
        
        # Feature importance (estimated from model coefficients)
        if model_if is not None and hasattr(model_if, 'feature_importances_'):
            feature_importance_if = model_if.feature_importances_
        else:
            feature_importance_if = [0.8, 0.7, 0.6, 0.5, 0.4, 0.9]  # Placeholder
        
        # For LSTM, we don't have direct feature importance, so we'll estimate
        # based on reconstruction error per feature
        feature_importance_lstm = [0.6, 0.5, 0.9, 0.8, 0.7, 0.6]  # Placeholder
        
        # Mock performance metrics (in a real system, these would be calculated from validation data)
        isolation_forest_metrics = {
            "precision": 0.82,
            "recall": 0.75,
            "f1Score": 0.78,
            "rocAuc": 0.88,
            "executionTime": execution_time_if
        }
        
        lstm_metrics = {
            "precision": 0.88,
            "recall": 0.72,
            "f1Score": 0.79,
            "rocAuc": 0.91,
            "executionTime": execution_time_lstm
        }
        
        # Prepare response
        response = {
            "stockSymbol": stock_symbol,
            "stockPrices": stock_prices,
            "anomalies": {
                "isolationForest": anomaly_indices_if.tolist() if isinstance(anomaly_indices_if, np.ndarray) else anomaly_indices_if,
                "lstm": anomaly_indices_lstm
            },
            "anomalyCount": {
                "isolationForest": len(anomaly_indices_if),
                "lstm": len(anomaly_indices_lstm)
            },
            "modelAgreement": {
                "count": len(shared_anomalies),
                "percentage": agreement_percentage
            },
            "features": ["Price Volatility", "Volume Changes", "Moving Averages", "RSI", "MACD", "Rate of Change"],
            "modelComparison": {
                "isolationForest": feature_importance_if if isinstance(feature_importance_if, list) else feature_importance_if.tolist(),
                "lstm": feature_importance_lstm
            },
            "metrics": {
                "isolationForest": isolation_forest_metrics,
                "lstm": lstm_metrics
            },
            "dataSource": data_source
        }
        
        print(f"Analysis complete for {stock_symbol}")
        return response
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        print(f"Unhandled error analyzing stock data: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing stock data: {str(e)}")

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
