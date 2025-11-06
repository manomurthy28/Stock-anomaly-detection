import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Bidirectional, BatchNormalization
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import time
from alpha_vantage_fetcher import get_stock_data

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


def compute_bollinger_bands(data, window=20, num_std=2):
    """Compute Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band


def compute_atr(high, low, close, window=14):
    """Compute Average True Range (ATR)"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def compute_stochastic_oscillator(high, low, close, k_window=14, d_window=3):
    """Compute Stochastic Oscillator"""
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + epsilon))
    # %D = 3-day SMA of %K
    d = k.rolling(window=d_window).mean()
    return k, d

# --------------------------------------
# Function to Create Sequences for LSTM
# --------------------------------------
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
    return np.array(sequences)

# --------------------------------------
# Build Enhanced LSTM Autoencoder Model
# --------------------------------------
def build_lstm_autoencoder(seq_length, n_features):
    model = Sequential([
        # Encoder
        Bidirectional(LSTM(128, activation='relu', return_sequences=True), 
                     input_shape=(seq_length, n_features)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, activation='relu', return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Bottleneck
        Dense(32, activation='relu'),
        BatchNormalization(),
        
        # Decoder
        RepeatVector(seq_length),
        
        Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(128, activation='relu', return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output
        TimeDistributed(Dense(n_features))
    ])
    
    # Use a more sophisticated optimizer with learning rate scheduling
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='mse'
    )
    return model

# --------------------------------------
# Detect Anomalies Using LSTM Autoencoder
# --------------------------------------
def detect_anomaly_lstm(stock_symbol):
    try:
        print(f"\nFetching stock data for LSTM analysis of {stock_symbol} using Alpha Vantage...")
        
        # Get data from Alpha Vantage
        stock_data = get_stock_data(stock_symbol)
        
        if stock_data is None or stock_data.empty:
            print(f"\n❌ LSTM: No data found for {stock_symbol}")
            return None, None
            
        print(f"LSTM: Successfully fetched data for {stock_symbol}. Shape: {stock_data.shape}")

        # Define epsilon to avoid division by zero
        epsilon = 1e-10

        # Enhanced Feature Engineering
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
        
        print(f"LSTM: Data shape after feature engineering: {stock_data.shape}")

        # Select Relevant Features
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
                print(f"LSTM Warning: Feature {feature} contains infinity or extreme values. Fixing...")
                # Replace with median or cap at reasonable values
                median_val = stock_data[feature].median()
                stock_data[feature] = stock_data[feature].replace([np.inf, -np.inf], median_val)
                # Cap extreme values at 1000 times the median (absolute value)
                cap_value = 1000 * abs(median_val) if median_val != 0 else 1000
                stock_data[feature] = stock_data[feature].clip(-cap_value, cap_value)
        
        data = stock_data[features].values
        
        # Final check for any remaining NaN or infinite values
        if np.isnan(data).any() or np.isinf(data).any():
            print("LSTM: Data still contains NaN or infinite values. Replacing with zeros...")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use RobustScaler for better handling of outliers
        try:
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Final check after scaling
            if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
                print("LSTM: Scaled data contains NaN or infinite values. Replacing with zeros...")
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
                
        except Exception as e:
            print(f"LSTM: Error during scaling: {e}")
            # If scaling fails, use the original data with simple normalization
            print("LSTM: Falling back to simple normalization...")
            # Simple normalization: subtract mean and divide by std for each column
            means = np.nanmean(data, axis=0)
            stds = np.nanstd(data, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
            scaled_data = (data - means) / stds
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Create Sequences for LSTM
        seq_length = 15  # Increased sequence length for better pattern recognition
        sequences = create_sequences(scaled_data, seq_length)
        
        print(f"LSTM: Created {len(sequences)} sequences with shape {sequences.shape}")

        # Train-Test Split
        train_size = int(len(sequences) * 0.8)
        train_data, test_data = sequences[:train_size], sequences[train_size:]
        
        if len(train_data) == 0 or len(test_data) == 0:
            print("LSTM: Not enough data for training and testing")
            return None, None

        # Build and Train Model
        print("LSTM: Building and training model...")
        model = build_lstm_autoencoder(seq_length, len(features))
        
        # Use early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Use learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model with more epochs and callbacks
        history = model.fit(
            train_data, train_data,
            epochs=50,  # Increased epochs for better learning
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        # Predict Reconstruction Error
        print("LSTM: Calculating reconstruction error...")
        train_pred = model.predict(train_data)
        test_pred = model.predict(test_data)
        
        # Calculate MSE for each sample and each feature
        train_mse = np.mean(np.power(train_data - train_pred, 2), axis=(1,2))
        test_mse = np.mean(np.power(test_data - test_pred, 2), axis=(1,2))
        
        # Calculate feature-wise reconstruction error for feature importance
        feature_mse = np.mean(np.power(test_data - test_pred, 2), axis=(0,1))
        feature_importance = feature_mse / np.sum(feature_mse)
        
        # Calculate threshold based on training data
        # Using a lower percentile (90th instead of 95th) to catch more anomalies
        # This will reduce false negatives at the cost of potentially more false positives
        threshold = np.percentile(train_mse, 90)
        
        # Detect anomalies in test data
        anomalies = (test_mse > threshold).astype(int)

        # Map anomalies back to original data indices
        anomaly_indices = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                # Add seq_length to account for the offset in the test data
                anomaly_indices.append(train_size + i + seq_length)
        
        print(f"LSTM: Detected {len(anomaly_indices)} anomalies")
        
        # Add post-processing to reduce false negatives
        # Look for consecutive days with high reconstruction error (but below threshold)
        near_threshold = threshold * 0.8  # 80% of the threshold
        for i in range(len(test_mse) - 2):
            # If we have 3 consecutive days with high error (but below threshold)
            if (test_mse[i] > near_threshold and 
                test_mse[i+1] > near_threshold and 
                test_mse[i+2] > near_threshold and
                not (i in anomalies or i+1 in anomalies or i+2 in anomalies)):
                
                # Mark the middle day as an anomaly
                idx = train_size + (i+1) + seq_length
                if idx not in anomaly_indices:
                    anomaly_indices.append(idx)
        
        # Also look for days with sudden spikes in reconstruction error
        for i in range(1, len(test_mse)):
            error_change = test_mse[i] / (test_mse[i-1] + epsilon)
            if error_change > 3.0 and test_mse[i] > near_threshold and i not in anomalies:
                idx = train_size + i + seq_length
                if idx not in anomaly_indices:
                    anomaly_indices.append(idx)
        
        # Sort the anomaly indices
        anomaly_indices.sort()
        
        print(f"LSTM: After post-processing, detected {len(anomaly_indices)} anomalies")

        return stock_data, anomaly_indices

    except Exception as e:
        print(f"\n❌ Error in detect_anomaly_lstm: {e}")
        import traceback
        traceback.print_exc()
        return None, None
