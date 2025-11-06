import { generateFallbackStockData } from "./fallback-analysis"

export async function fetchStockData(symbol: string, useFallback = false) {
  // If fallback is explicitly requested, use it immediately
  if (useFallback) {
    console.log(`Using fallback data for ${symbol} as requested`)
    return generateFallbackStockData(symbol)
  }

  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
    console.log(`Fetching stock data for ${symbol} from ${apiUrl}/analyze/${symbol}`)

    const response = await fetch(`${apiUrl}/analyze/${symbol}`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      signal: AbortSignal.timeout(30000), // 30 second timeout
    })

    if (!response.ok) {
      let errorMessage = `API error: ${response.status}`
      try {
        const errorData = await response.json()
        errorMessage += ` - ${errorData.detail || errorData.message || "Unknown error"}`
      } catch (parseError) {
        errorMessage += " - Could not parse error response"
      }

      throw new Error(errorMessage)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error("Error fetching stock data:", error)
    throw error
  }
}

export async function fetchAnomaliesByDate(startDate: Date, endDate: Date, model?: string, stockSymbol?: string) {
  try {
    const params = new URLSearchParams({
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
    })

    if (model) {
      params.append("model", model)
    }

    if (stockSymbol) {
      params.append("stockSymbol", stockSymbol)
    }

    const response = await fetch(`/api/anomalies/by-date?${params.toString()}`)

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.error || "Failed to fetch anomalies")
    }

    const data = await response.json()
    return data.anomalies
  } catch (error) {
    console.error("Error fetching anomalies by date:", error)
    throw error
  }
}
