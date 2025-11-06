from alpha_vantage_fetcher import get_stock_data
from isolation_forest import detect_anomaly_isolation_forest
from lstm_autoencoder import detect_anomaly_lstm
from model_validation import validate_anomaly_detection, determine_actual_events
import numpy as np
import sys

def test_confusion_matrix(symbol):
    """
    Test the confusion matrix calculation for a given symbol
    """
    print(f"Testing confusion matrix for symbol: {symbol}")
    
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
    
    # Calculate shared anomalies
    shared_indices = [idx for idx in anomaly_indices_if if idx in anomaly_indices_lstm]
    agreement_percentage = round(
        (len(shared_indices) / ((len(anomaly_indices_if) + len(anomaly_indices_lstm)) / 2)) * 100
    ) if (len(anomaly_indices_if) + len(anomaly_indices_lstm)) > 0 else 0
    
    print(f"\nModel Agreement: {agreement_percentage}% ({len(shared_indices)} shared anomalies)")
    
    # Validate model predictions
    print("Calculating confusion matrix...")
    if_validation = validate_anomaly_detection(if_predictions, actual_events)
    lstm_validation = validate_anomaly_detection(lstm_predictions, actual_events)
    
    # Print confusion matrix
    print("\nIsolation Forest Confusion Matrix:")
    print(f"True Positives: {if_validation['truePositives']}")
    print(f"False Positives: {if_validation['falsePositives']}")
    print(f"True Negatives: {if_validation['trueNegatives']}")
    print(f"False Negatives: {if_validation['falseNegatives']}")
    
    print("\nLSTM Autoencoder Confusion Matrix:")
    print(f"True Positives: {lstm_validation['truePositives']}")
    print(f"False Positives: {lstm_validation['falsePositives']}")
    print(f"True Negatives: {lstm_validation['trueNegatives']}")
    print(f"False Negatives: {lstm_validation['falseNegatives']}")
    
    # Print metrics
    print("\nIsolation Forest Metrics:")
    print(f"Accuracy: {if_validation['accuracy']:.4f}")
    print(f"Precision: {if_validation['precision']:.4f}")
    print(f"Recall: {if_validation['recall']:.4f}")
    print(f"F1 Score: {if_validation['f1Score']:.4f}")
    
    print("\nLSTM Autoencoder Metrics:")
    print(f"Accuracy: {lstm_validation['accuracy']:.4f}")
    print(f"Precision: {lstm_validation['precision']:.4f}")
    print(f"Recall: {lstm_validation['recall']:.4f}")
    print(f"F1 Score: {lstm_validation['f1Score']:.4f}")
    
    return True

if __name__ == "__main__":
    # Get symbol from command line argument or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    # Test confusion matrix
    success = test_confusion_matrix(symbol)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
