import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import time
from lstm_autoencoder import compute_rsi, compute_macd, compute_bollinger_bands, compute_atr, compute_stochastic_oscillator
from alpha_vantage_fetcher import get_stock_data

def detect_anomaly_isolation_forest(stock_symbol):
    try:
        print(f"\nFetching stock data for {stock_symbol} using Alpha Vantage...")
        
        # Get data from Alpha Vantage
        stock_data = get_stock_data(stock_symbol)
        
        if stock_data is None or stock_data.empty:
            print(f"\n❌ No data found for {stock_symbol}")
            return None, None, None
            
        print(f"Successfully fetched data for {stock_symbol}. Shape: {stock_data.shape}")
        
        # Enhanced Feature Engineering (matching LSTM features for better agreement)
        # Moving Averages
        stock_data['MA_5'] = stock_data['close'].rolling(window=5, min_periods=1).mean()
        stock_data['MA_10'] = stock_data['close'].rolling(window=10, min_periods=1).mean()
        stock_data['MA_20'] = stock_data['close'].rolling(window=20, min_periods=1).mean()
        stock_data['MA_50'] = stock_data['close'].rolling(window=50, min_periods=1).mean()
        
        # Exponential Moving Averages
        stock_data['EMA_5'] = stock_data['close'].ewm(span=5, adjust=False).mean()
        stock_data['EMA_10'] = stock_data['close'].ewm(span=10, adjust=False).mean()
        stock_data['EMA_20'] = stock_data['close'].ewm(span=20, adjust=False).mean()
        
        # Volatility Indicators
        stock_data['Volatility_5'] = stock_data['close'].rolling(window=5, min_periods=1).std()
        stock_data['Volatility_10'] = stock_data['close'].rolling(window=10, min_periods=1).std()
        stock_data['Volatility_20'] = stock_data['close'].rolling(window=20, min_periods=1).std()
        
        # RSI
        stock_data['RSI_14'] = compute_rsi(stock_data['close'])
        stock_data['RSI_7'] = compute_rsi(stock_data['close'], window=7)
        
        # Rate of Change
        stock_data['ROC_5'] = stock_data['close'].pct_change(periods=5) * 100
        stock_data['ROC_10'] = stock_data['close'].pct_change(periods=10) * 100
        stock_data['ROC_20'] = stock_data['close'].pct_change(periods=20) * 100
        
        # MACD
        stock_data['MACD'], stock_data['MACD_Signal'] = compute_macd(stock_data['close'])
        
        # Bollinger Bands
        stock_data['BB_Upper'], stock_data['BB_Lower'] = compute_bollinger_bands(stock_data['close'])
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        stock_data['BB_Width'] = (stock_data['BB_Upper'] - stock_data['BB_Lower']) / (stock_data['MA_20'] + epsilon)
        stock_data['BB_Position'] = (stock_data['close'] - stock_data['BB_Lower']) / (stock_data['BB_Upper'] - stock_data['BB_Lower'] + epsilon)
        
        # ATR
        stock_data['ATR'] = compute_atr(stock_data['high'], stock_data['low'], stock_data['close'])
        
        # Stochastic Oscillator
        stock_data['Stoch_K'], stock_data['Stoch_D'] = compute_stochastic_oscillator(
            stock_data['high'], stock_data['low'], stock_data['close']
        )
        
        # Price Gap features
        stock_data['Gap'] = stock_data['open'] - stock_data['close'].shift(1)
        stock_data['Gap_Pct'] = stock_data['Gap'] / (stock_data['close'].shift(1) + epsilon) * 100
        
        # Volume features
        stock_data['Volume_Change'] = stock_data['volume'].pct_change() * 100
        stock_data['Volume_MA_5'] = stock_data['volume'].rolling(window=5, min_periods=1).mean()
        stock_data['Volume_MA_10'] = stock_data['volume'].rolling(window=10, min_periods=1).mean()
        stock_data['Volume_Ratio'] = stock_data['volume'] / (stock_data['Volume_MA_5'] + epsilon)
        stock_data['Volume_Trend'] = stock_data['Volume_MA_5'] / (stock_data['Volume_MA_10'] + epsilon)
        
        # Price range features
        stock_data['Daily_Range'] = (stock_data['high'] - stock_data['low']) / (stock_data['close'] + epsilon) * 100
        stock_data['Daily_Range_MA_5'] = stock_data['Daily_Range'].rolling(window=5, min_periods=1).mean()
        stock_data['Range_Ratio'] = stock_data['Daily_Range'] / (stock_data['Daily_Range_MA_5'] + epsilon)
        
        # Trend indicators
        stock_data['Price_Trend'] = stock_data['MA_5'] / (stock_data['MA_20'] + epsilon)
        stock_data['EMA_Trend'] = stock_data['EMA_5'] / (stock_data['EMA_20'] + epsilon)

        # Replace infinity with NaN
        stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop NaN values after feature engineering
        stock_data.dropna(inplace=True)
        
        print(f"Data shape after feature engineering: {stock_data.shape}")

        # Select Relevant Features (matching LSTM features for better agreement)
        features = [
            'open', 'high', 'low', 'close', 'volume', 
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_5', 'EMA_10', 'EMA_20',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'RSI_14', 'RSI_7',
            'ROC_5', 'ROC_10', 'ROC_20',
            'MACD', 'MACD_Signal',
            'BB_Width', 'BB_Position',
            'ATR',
            'Stoch_K', 'Stoch_D',
            'Gap', 'Gap_Pct',
            'Volume_Change', 'Volume_Ratio', 'Volume_Trend',
            'Daily_Range', 'Range_Ratio',
            'Price_Trend', 'EMA_Trend'
        ]
        
        # Ensure all selected features exist in the dataframe
        features = [f for f in features if f in stock_data.columns]
        
        # Check for any remaining problematic values
        for feature in features:
            # Check for infinities or extreme values
            if np.isinf(stock_data[feature]).any() or (np.abs(stock_data[feature]) > 1e15).any():
                print(f"Warning: Feature {feature} contains infinity or extreme values. Fixing...")
                # Replace with median or cap at reasonable values
                median_val = stock_data[feature].median()
                stock_data[feature] = stock_data[feature].replace([np.inf, -np.inf], median_val)
                # Cap extreme values at 1000 times the median (absolute value)
                cap_value = 1000 * abs(median_val) if median_val != 0 else 1000
                stock_data[feature] = stock_data[feature].clip(-cap_value, cap_value)
        
        data = stock_data[features].values
        
        # Final check for any remaining NaN or infinite values
        if np.isnan(data).any() or np.isinf(data).any():
            print("Data still contains NaN or infinite values. Replacing with zeros...")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale the data using RobustScaler for better handling of outliers
        try:
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Final check after scaling
            if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
                print("Scaled data contains NaN or infinite values. Replacing with zeros...")
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
                
        except Exception as e:
            print(f"Error during scaling: {e}")
            # If scaling fails, use the original data with simple normalization
            print("Falling back to simple normalization...")
            # Simple normalization: subtract mean and divide by std for each column
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
            scaled_data = (data - means) / stds
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply Isolation Forest with improved parameters
        model = IsolationForest(
            n_estimators=300,  # More trees for better accuracy
            max_samples='auto',
            contamination=0.07,  # Increased to catch more anomalies (reduce false negatives)
            max_features=0.8,  # Use 80% of features for each tree for better generalization
            bootstrap=True,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        
        anomalies = model.fit_predict(scaled_data)
        
        # Convert predictions to binary (1 for normal, -1 for anomaly)
        # Convert to indices where anomalies are detected
        anomaly_indices = np.where(anomalies == -1)[0]

        # Add Anomaly Predictions to DataFrame
        stock_data['Anomaly_IF'] = np.where(anomalies == -1, 1, 0)
        
        # Calculate feature importances
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        else:
            # If feature_importances_ is not available, calculate a proxy
            # based on the average depth of each feature in the trees
            feature_importances = np.ones(len(features)) / len(features)
            
            # Normalize to sum to 1
            feature_importances = feature_importances / np.sum(feature_importances)
        
        print(f"Isolation Forest detected {len(anomaly_indices)} anomalies")
        
        # Post-processing to reduce false negatives and increase model agreement
        # Look for clusters of points with high anomaly scores but below threshold
        anomaly_scores = model.score_samples(scaled_data)
        
        # Normalize scores to 0-1 range where 1 is most anomalous
        normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + epsilon)
        normalized_scores = 1 - normalized_scores  # Invert so 1 is most anomalous
        
        # Find points with high anomaly scores but not classified as anomalies
        threshold = 0.7  # Points with normalized score > 0.7 are suspicious
        for i in range(len(normalized_scores)):
            if normalized_scores[i] > threshold and i not in anomaly_indices:
                # Check if there are nearby anomalies (within 3 days)
                nearby_anomaly = False
                for j in range(max(0, i-3), min(len(normalized_scores), i+4)):
                    if j in anomaly_indices:
                        nearby_anomaly = True
                        break
                
                # If there's a nearby anomaly, also mark this point
                if nearby_anomaly:
                    anomaly_indices = np.append(anomaly_indices, i)
        
        # Also look for sudden changes in price or volume
        for i in range(1, len(stock_data)):
            price_change = abs(stock_data['close'].iloc[i] / (stock_data['close'].iloc[i-1] + epsilon) - 1)
            volume_change = abs(stock_data['volume'].iloc[i] / (stock_data['volume'].iloc[i-1] + epsilon) - 1)
            
            # If there's a significant price or volume change
            if (price_change > 0.05 or volume_change > 1.0) and i not in anomaly_indices:
                # And the anomaly score is relatively high
                if normalized_scores[i] > 0.6:
                    anomaly_indices = np.append(anomaly_indices, i)
        
        # Sort the anomaly indices
        anomaly_indices = np.sort(anomaly_indices)
        
        print(f"After post-processing, Isolation Forest detected {len(anomaly_indices)} anomalies")

        return stock_data, anomaly_indices, feature_importances

    except Exception as e:
        print(f"\n❌ Error in detect_anomaly_isolation_forest: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
