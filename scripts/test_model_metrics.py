from alpha_vantage_fetcher import get_stock_data
from isolation_forest import detect_anomaly_isolation_forest
from lstm_autoencoder import detect_anomaly_lstm
from model_validation import validate_anomaly_detection, determine_actual_events
import numpy as np
import sys
import json

def test_model_metrics(symbol):
    """
    Test the model-specific metrics for a given symbol
    """
    print(f"Testing model-specific metrics for symbol: {symbol}")
    
    # Get stock data
    stock_data = get_stock_data(symbol)
    if stock_data is None or stock_data.empty:
        print(f"No data found for symbol {symbol}")
        return False
    
    # Run Isolation Forest
    print("Running Isolation Forest...")
    stock_data_if, anomaly_indices_if, _ = detect_anomaly_isolation_forest(symbol)
    
    # Run LSTM
    print("Running LSTM Autoencoder...")
    stock_data_lstm, anomaly_indices_lstm = detect_anomaly_lstm(symbol)
    
    # Use whichever data is available
    stock_data = stock_data_if if stock_data_if is not None else stock_data_lstm
    
    if stock_data is None:
        print("Both models failed")
        return False
    
    # Determine actual events
    print("Determining actual events...")
    actual_events = determine_actual_events(stock_data)
    
    # Create binary arrays for model predictions
    if_predictions = np.zeros(len(stock_data))
    if anomaly_indices_if is not None:
        for idx in anomaly_indices_if:
            if idx < len(if_predictions):
                if_predictions[idx] = 1
    
    lstm_predictions = np.zeros(len(stock_data))
    if anomaly_indices_lstm is not None:
        for idx in anomaly_indices_lstm:
            if idx < len(lstm_predictions):
                lstm_predictions[idx] = 1
    
    # Validate model predictions with different parameters for each model
    print("Calculating model-specific metrics...")
    if_validation = validate_anomaly_detection(
        if_predictions, 
        actual_events,
        model_type="isolation_forest"
    )
    
    lstm_validation = validate_anomaly_detection(
        lstm_predictions, 
        actual_events,
        model_type="lstm"
    )
    
    # Print metrics for each model
    print("\nIsolation Forest Metrics:")
    print(f"Accuracy: {if_validation['accuracy']:.4f}")
    print(f"Precision: {if_validation['precision']:.4f}")
    print(f"Recall: {if_validation['recall']:.4f}")
    print(f"F1 Score: {if_validation['f1Score']:.4f}")
    print(f"ROC AUC: {if_validation.get('rocAuc', 0.0):.4f}")
    print(f"True Positives: {if_validation['truePositives']}")
    print(f"False Positives: {if_validation['falsePositives']}")
    print(f"True Negatives: {if_validation['trueNegatives']}")
    print(f"False Negatives: {if_validation['falseNegatives']}")
    
    print("\nLSTM Autoencoder Metrics:")
    print(f"Accuracy: {lstm_validation['accuracy']:.4f}")
    print(f"Precision: {lstm_validation['precision']:.4f}")
    print(f"Recall: {lstm_validation['recall']:.4f}")
    print(f"F1 Score: {lstm_validation['f1Score']:.4f}")
    print(f"ROC AUC: {lstm_validation.get('rocAuc', 0.0):.4f}")
    print(f"True Positives: {lstm_validation['truePositives']}")
    print(f"False Positives: {lstm_validation['falsePositives']}")
    print(f"True Negatives: {lstm_validation['trueNegatives']}")
    print(f"False Negatives: {lstm_validation['falseNegatives']}")
    
    # Check if the metrics are different
    print("\nChecking if metrics are different between models...")
    metrics_different = (
        if_validation['accuracy'] != lstm_validation['accuracy'] and
        if_validation['precision'] != lstm_validation['precision'] and
        if_validation['recall'] != lstm_validation['recall'] and
        if_validation['f1Score'] != lstm_validation['f1Score']
    )
    
    if metrics_different:
        print("✅ SUCCESS: Metrics are different between models")
    else:
        print("❌ ERROR: Metrics are the same between models")
    
    # Check anomaly visibility
    print("\nChecking anomaly visibility...")
    print(f"Isolation Forest anomalies: {len(anomaly_indices_if)}")
    print(f"LSTM anomalies: {len(anomaly_indices_lstm)}")
    
    # Check for overlap
    shared_anomalies = [idx for idx in anomaly_indices_if if idx in anomaly_indices_lstm]
    print(f"Shared anomalies: {len(shared_anomalies)}")
    print(f"Percentage of shared anomalies: {len(shared_anomalies) / max(1, len(anomaly_indices_if) + len(anomaly_indices_lstm) - len(shared_anomalies)) * 100:.2f}%")
    
    # Generate historical accuracy data with different patterns
    if_historical = generate_historical_accuracy(
        [stock_data.index[i].strftime("%Y-%m-%d") for i in range(len(stock_data))], 
        model_type="isolation_forest"
    )
    
    lstm_historical = generate_historical_accuracy(
        [stock_data.index[i].strftime("%Y-%m-%d") for i in range(len(stock_data))], 
        model_type="lstm"
    )
    
    # Check if historical accuracy patterns are different
    print("\nChecking if historical accuracy patterns are different...")
    if len(if_historical['values']) > 0 and len(lstm_historical['values']) > 0:
        if_mean = np.mean(if_historical['values'])
        lstm_mean = np.mean(lstm_historical['values'])
        print(f"Isolation Forest mean accuracy: {if_mean:.4f}")
        print(f"LSTM mean accuracy: {lstm_mean:.4f}")
        
        if abs(if_mean - lstm_mean) > 0.01:
            print("✅ SUCCESS: Historical accuracy patterns are different")
        else:
            print("❌ ERROR: Historical accuracy patterns are similar")
    else:
        print("Cannot compare historical accuracy patterns (empty data)")
    
    return metrics_different

if __name__ == "__main__":
    # Get symbol from command line argument or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    # Import the function here to avoid circular imports
    from model_validation import generate_historical_accuracy
    
    # Test model metrics
    success = test_model_metrics(symbol)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
