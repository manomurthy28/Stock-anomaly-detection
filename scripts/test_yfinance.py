import yfinance as yf
import pandas as pd
import sys

def test_yfinance(symbol):
    """
    Test if yfinance can fetch data for a given symbol
    """
    print(f"Testing yfinance for symbol: {symbol}")
    
    try:
        # Try to fetch data
        data = yf.download(symbol, period="1mo", progress=False)
        
        # Check if data is empty
        if data.empty:
            print(f"No data found for symbol {symbol}")
            return False
        
        # Print some information about the data
        print(f"Successfully fetched data for {symbol}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"First few rows:")
        print(data.head())
        
        return True
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False

if __name__ == "__main__":
    # Get symbol from command line argument or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    # Test yfinance
    success = test_yfinance(symbol)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
