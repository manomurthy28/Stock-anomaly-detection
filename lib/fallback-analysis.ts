// Fallback data generator for when the backend API is unavailable
import { subDays, format } from "date-fns"

// Generate synthetic stock data for fallback mode
export function generateFallbackStockData(symbol: string) {
  console.log(`Generating fallback data for ${symbol}`)

  // Check if it's an Indian stock
  const isIndianStock = symbol.endsWith(".NS") || symbol.endsWith(".BO")

  // Generate base price and volatility based on the symbol
  // This is just for demonstration - in a real app, you'd use more sophisticated methods
  const basePrice = getBasePrice(symbol)
  const volatility = isIndianStock ? 0.015 : 0.01 // Indian stocks are slightly more volatile in our simulation

  // Generate 252 days of data (approximately 1 trading year)
  const today = new Date()
  const stockPrices = []

  for (let i = 251; i >= 0; i--) {
    const date = subDays(today, i)
    const dateStr = format(date, "yyyy-MM-dd")

    // Simple random walk with drift
    const randomChange = (Math.random() - 0.5) * 2 * volatility
    const price = i === 251 ? basePrice : stockPrices[stockPrices.length - 1].price * (1 + randomChange)

    stockPrices.push({
      date: dateStr,
      price: Number.parseFloat(price.toFixed(2)),
    })
  }

  // Generate anomalies - approximately 5% of days will be anomalies
  const totalDays = stockPrices.length
  const anomalyCount = Math.floor(totalDays * 0.05)

  // For Isolation Forest
  const anomaliesIF = generateAnomalyIndices(totalDays, anomalyCount)

  // For LSTM - some overlap with IF, some unique
  const sharedCount = Math.floor(anomalyCount * 0.6) // 60% shared
  const uniqueLSTMCount = anomalyCount - sharedCount

  const sharedAnomalies = anomaliesIF.slice(0, sharedCount)
  const uniqueLSTMAnomalies = generateUniqueAnomalyIndices(totalDays, uniqueLSTMCount, anomaliesIF)
  const anomaliesLSTM = [...sharedAnomalies, ...uniqueLSTMAnomalies].sort((a, b) => a - b)

  // Calculate model agreement
  const sharedCount2 = sharedAnomalies.length
  const agreementPercentage = Math.round((sharedCount2 / ((anomaliesIF.length + anomaliesLSTM.length) / 2)) * 100)

  // Generate feature importance
  const featureImportanceIF = [0.85, 0.75, 0.9, 0.8, 0.7, 0.82]
  const featureImportanceLSTM = [0.78, 0.88, 0.75, 0.92, 0.83, 0.7]

  // Generate metrics
  const isolationForestMetrics = {
    precision: 0.82,
    recall: 0.75,
    f1Score: 0.78,
    rocAuc: 0.88,
    executionTime: 0.45,
  }

  const lstmMetrics = {
    precision: 0.88,
    recall: 0.72,
    f1Score: 0.79,
    rocAuc: 0.91,
    executionTime: 1.23,
  }

  // Generate dates for historical accuracy
  const historicalDates = stockPrices.filter((_, i) => i % 20 === 0).map((sp) => sp.date)

  // Generate validation metrics with the correct structure for ModelAccuracy component
  const isolationForestValidation = {
    truePositives: 18,
    falsePositives: 4,
    trueNegatives: 220,
    falseNegatives: 10,
    accuracy: 0.85,
    precision: 0.82,
    recall: 0.75,
    f1Score: 0.78,
    historicalAccuracy: {
      dates: historicalDates,
      values: historicalDates.map(() => 0.7 + Math.random() * 0.25), // Random accuracy between 70% and 95%
    },
  }

  const lstmValidation = {
    truePositives: 20,
    falsePositives: 3,
    trueNegatives: 222,
    falseNegatives: 7,
    accuracy: 0.88,
    precision: 0.88,
    recall: 0.72,
    f1Score: 0.79,
    historicalAccuracy: {
      dates: historicalDates,
      values: historicalDates.map(() => 0.7 + Math.random() * 0.25), // Random accuracy between 70% and 95%
    },
  }

  // Return the complete fallback data
  return {
    stockSymbol: symbol,
    stockPrices: stockPrices,
    anomalies: {
      isolationForest: anomaliesIF,
      lstm: anomaliesLSTM,
    },
    anomalyCount: {
      isolationForest: anomaliesIF.length,
      lstm: anomaliesLSTM.length,
    },
    modelAgreement: {
      count: sharedCount2,
      percentage: agreementPercentage,
    },
    features: ["Price Volatility", "Volume Changes", "Moving Averages", "RSI", "MACD", "Rate of Change"],
    modelComparison: {
      isolationForest: featureImportanceIF,
      lstm: featureImportanceLSTM,
    },
    metrics: {
      isolationForest: isolationForestMetrics,
      lstm: lstmMetrics,
    },
    validation: {
      isolationForest: isolationForestValidation,
      lstm: lstmValidation,
    },
    isFallbackData: true,
  }
}

// Helper function to get a base price for a stock symbol
function getBasePrice(symbol: string): number {
  // Common US stocks
  const stockPrices: Record<string, number> = {
    AAPL: 175.5,
    MSFT: 350.2,
    GOOGL: 125.3,
    AMZN: 130.4,
    META: 310.75,
    TSLA: 180.6,
    NVDA: 420.3,
    JPM: 145.8,
    V: 240.5,
    WMT: 65.25,
    // Indian stocks
    "SBIN.NS": 750.25,
    "TCS.NS": 3500.75,
    "RELIANCE.NS": 2400.5,
    "INFY.NS": 1450.3,
    "HDFCBANK.NS": 1650.4,
    "ICICIBANK.NS": 950.6,
    "HINDUNILVR.NS": 2500.25,
    "BHARTIARTL.NS": 850.75,
    "KOTAKBANK.NS": 1750.8,
    "ITC.NS": 450.25,
  }

  // If we have a predefined price, use it, otherwise generate a random one
  if (symbol in stockPrices) {
    return stockPrices[symbol]
  }

  // For Indian stocks, generate a higher base price (typical for Indian markets)
  if (symbol.endsWith(".NS") || symbol.endsWith(".BO")) {
    return 500 + Math.random() * 2000
  }

  // For other stocks, generate a random price between $10 and $500
  return 10 + Math.random() * 490
}

// Generate random anomaly indices
function generateAnomalyIndices(totalDays: number, count: number): number[] {
  const indices: number[] = []
  while (indices.length < count) {
    const idx = Math.floor(Math.random() * totalDays)
    if (!indices.includes(idx)) {
      indices.push(idx)
    }
  }
  return indices.sort((a, b) => a - b)
}

// Generate unique anomaly indices that don't overlap with existing ones
function generateUniqueAnomalyIndices(totalDays: number, count: number, existing: number[]): number[] {
  const indices: number[] = []
  while (indices.length < count) {
    const idx = Math.floor(Math.random() * totalDays)
    if (!indices.includes(idx) && !existing.includes(idx)) {
      indices.push(idx)
    }
  }
  return indices.sort((a, b) => a - b)
}
