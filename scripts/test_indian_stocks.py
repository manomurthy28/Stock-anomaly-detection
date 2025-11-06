import yfinance as yf
import pandas as pd
import sys
import requests
import time
import json

def test_indian_stocks():
    """
    Test fetching data for Indian stock symbols with multiple approaches
    """
    # List of common Indian stock symbols
    indian_symbols = [
        "SBIN.NS",    # State Bank of India
        "TCS.NS",     # Tata Consultancy Services
        "RELIANCE.NS", # Reliance Industries
        "INFY.NS",    # Infosys
        "HDFCBANK.NS" # HDFC Bank
    ]
    
    print("Testing Indian stock symbols with multiple approaches...")
    
    # Try alternative formats
    alternative_formats = {
        "SBIN.NS": ["SBIN.BO", "SBIN", "SBI.NS"],
        "TCS.NS": ["TCS.BO", "TCS", "TATACONSUM.NS"],
        "RELIANCE.NS": ["RELIANCE.BO", "RELIANCE", "RELI.NS"],
        "INFY.NS": ["INFY.BO", "INFY", "INFY.N"],
        "HDFCBANK.NS": ["HDFCBANK.BO", "HDFCBANK", "HDB"]
    }
    
    successful_symbols = {}
    
    for symbol in indian_symbols:
        print(f"\n===== Testing {symbol} =====")
        
        # Try with yfinance
        success = test_with_yfinance(symbol)
        if success:
            successful_symbols[symbol] = symbol
            continue
            
        # Try alternative formats
        if symbol in alternative_formats:
            for alt_symbol in alternative_formats[symbol]:
                print(f"\nTrying alternative format: {alt_symbol}")
                success = test_with_yfinance(alt_symbol)
                if success:
                    successful_symbols[symbol] = alt_symbol
                    break
        
        # If still not successful, try direct Yahoo Finance API
        if symbol not in successful_symbols:
            print(f"\nTrying direct Yahoo Finance API for {symbol}")
            success = test_with_direct_yahoo(symbol)
            if success:
                successful_symbols[symbol] = symbol
    
    # Summary
    print("\n===== SUMMARY =====")
    if successful_symbols:
        print(f"Successfully fetched data for {len(successful_symbols)}/{len(indian_symbols)} symbols:")
        for original, working in successful_symbols.items():
            print(f"  - {original} (working as: {working})")
    else:
        print("❌ Could not fetch data for any Indian stock symbols")
        
    # Recommendations
    print("\n===== RECOMMENDATIONS =====")
    if not successful_symbols:
        print("1. Try using Alpha Vantage API instead of Yahoo Finance")
        print("2. Consider using a paid API service for Indian stocks")
        print("3. Check network connectivity and proxy settings")
        print("4. Try using BSE symbols (.BO) instead of NSE symbols (.NS)")
    else:
        print("Use the following working symbols in your application:")
        for original, working in successful_symbols.items():
            print(f"  - Use '{working}' instead of '{original}'")
    
    return len(successful_symbols) > 0

def test_with_yfinance(symbol):
    """Test fetching data with yfinance"""
    try:
        # Try to fetch data
        data = yf.download(symbol, period="1mo", progress=False)
        
        # Check if data is empty
        if data.empty:
            print(f"❌ No data found for symbol {symbol}")
            return False
        
        # Print some information about the data
        print(f"✅ Successfully fetched data for {symbol}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"First few rows:")
        print(data.head(3))
        return True
    
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return False

def test_with_direct_yahoo(symbol):
    """Test fetching data directly from Yahoo Finance API"""
    try:
        # Construct Yahoo Finance API URL
        query = {
            "symbols": symbol,
            "range": "1mo",
            "interval": "1d",
            "indicators": "quote",
            "includeTimestamps": True,
            "includePrePost": False,
            "corsDomain": "finance.yahoo.com"
        }
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, params=query)
        
        if response.status_code != 200:
            print(f"❌ API returned status code {response.status_code}")
            return False
            
        data = response.json()
        
        # Check if we have valid data
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            print(f"❌ No data returned from Yahoo Finance API for {symbol}")
            return False
            
        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        
        if not timestamps:
            print(f"❌ No timestamps found in response for {symbol}")
            return False
            
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        
        # Check if we have price data
        if not quote.get("close"):
            print(f"❌ No price data found in response for {symbol}")
            return False
            
        print(f"✅ Successfully fetched data directly from Yahoo Finance API for {symbol}")
        print(f"Data points: {len(timestamps)}")
        print(f"Date range: {time.strftime('%Y-%m-%d', time.localtime(timestamps[0]))} to {time.strftime('%Y-%m-%d', time.localtime(timestamps[-1]))}")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching data directly from Yahoo Finance API for {symbol}: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    success = test_indian_stocks()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
