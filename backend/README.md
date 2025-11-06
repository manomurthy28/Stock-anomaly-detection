# Stock Anomaly Detection Backend

This is the Python backend for the Stock Anomaly Detection dashboard. It provides API endpoints for analyzing stock data and detecting anomalies using both Isolation Forest and LSTM Autoencoder models.

## Setup Instructions

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Start the server:
   \`\`\`bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   \`\`\`
   
## API Endpoints

- `GET /`: Check if the API is running
- `GET /analyze/{stock_symbol}`: Analyze stock data for the given symbol and return anomaly detection results

## Environment Variables

- `ALPHA_VANTAGE_API_KEY`: (Optional) API key for Alpha Vantage if you want to use it instead of yfinance

## Models Used

1. **Isolation Forest**: A tree-based algorithm that isolates outliers by randomly selecting features and splitting values
2. **LSTM Autoencoder**: A deep learning approach that uses Long Short-Term Memory networks to learn normal patterns and identify deviations
