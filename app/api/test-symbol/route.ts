import { type NextRequest, NextResponse } from "next/server"
import { logSystemEvent } from "@/lib/logger"

export async function GET(request: NextRequest) {
  try {
    // Get the symbol from the URL or query parameters
    const url = new URL(request.url)
    const symbol = url.searchParams.get("symbol") || ""

    if (!symbol) {
      return NextResponse.json({ available: false, message: "No symbol provided" }, { status: 400 })
    }

    console.log(`Testing symbol availability: ${symbol}`)
    await logSystemEvent("info", `Testing stock symbol: ${symbol}`)

    // Check if it's an Indian stock
    const isIndianStock = symbol.endsWith(".NS") || symbol.endsWith(".BO")

    if (isIndianStock) {
      console.log("Indian stock detected, using fallback mode")
      await logSystemEvent("info", `Using fallback mode for Indian stock: ${symbol}`)

      // For Indian stocks, return a fallback response
      return NextResponse.json({
        available: true,
        symbol: symbol,
        days_available: 252, // Typical trading days in a year
        date_range: {
          start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
          end: new Date().toISOString().split("T")[0],
        },
        is_indian_stock: true,
        message: "Using fallback data for Indian stock",
      })
    }

    // For non-Indian stocks, try to call the backend API
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
      const testUrl = `${apiUrl}/test-symbol/${symbol}`

      console.log(`Calling backend API: ${testUrl}`)
      await logSystemEvent("info", `Calling backend API for symbol: ${symbol}`)

      const response = await fetch(testUrl, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
        signal: AbortSignal.timeout(15000), // 15 second timeout
      })

      if (!response.ok) {
        console.error(`Backend API error: ${response.status}`)
        await logSystemEvent("error", `Backend API error for symbol ${symbol}: ${response.status}`)

        // Try to get more error details
        let errorDetail = ""
        try {
          const errorText = await response.text()
          console.error(`Error response: ${errorText}`)
          errorDetail = errorText.substring(0, 100)
        } catch (e) {
          errorDetail = "Could not read error response"
        }

        // For API errors, use a fallback for common US stocks
        const commonUsStocks = [
          "AAPL",
          "MSFT",
          "GOOGL",
          "AMZN",
          "META",
          "TSLA",
          "NVDA",
          "JPM",
          "V",
          "PG",
          "JNJ",
          "WMT",
          "DIS",
          "KO",
          "PEP",
          "NFLX",
          "ADBE",
          "CSCO",
          "INTC",
          "IBM",
        ]

        if (commonUsStocks.includes(symbol)) {
          console.log(`Using fallback for common US stock: ${symbol}`)
          await logSystemEvent("info", `Using fallback for common US stock: ${symbol}`)

          return NextResponse.json({
            available: true,
            symbol: symbol,
            days_available: 252,
            date_range: {
              start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
              end: new Date().toISOString().split("T")[0],
            },
            is_indian_stock: false,
            message: "Using fallback data for US stock",
          })
        }

        return NextResponse.json(
          {
            available: false,
            error: `API error: ${response.status} - ${errorDetail}`,
          },
          { status: 200 }, // Return 200 to the client but with error info
        )
      }

      const data = await response.json()
      return NextResponse.json(data)
    } catch (apiError) {
      console.error("Error calling backend API:", apiError)
      await logSystemEvent("error", `Error calling backend API for symbol ${symbol}: ${apiError}`)

      // For API errors, use a fallback for common US stocks
      const commonUsStocks = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "V",
        "PG",
        "JNJ",
        "WMT",
        "DIS",
        "KO",
        "PEP",
        "NFLX",
        "ADBE",
        "CSCO",
        "INTC",
        "IBM",
      ]

      if (commonUsStocks.includes(symbol)) {
        console.log(`Using fallback for common US stock: ${symbol}`)
        await logSystemEvent("info", `Using fallback for common US stock: ${symbol}`)

        return NextResponse.json({
          available: true,
          symbol: symbol,
          days_available: 252,
          date_range: {
            start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split("T")[0],
            end: new Date().toISOString().split("T")[0],
          },
          is_indian_stock: false,
          message: "Using fallback data for US stock",
        })
      }

      return NextResponse.json(
        {
          available: false,
          error: `Error calling backend API: ${apiError instanceof Error ? apiError.message : "Unknown error"}`,
        },
        { status: 200 }, // Return 200 to the client but with error info
      )
    }
  } catch (error) {
    console.error("Error in test-symbol route:", error)
    await logSystemEvent("error", `Error in test-symbol route: ${error}`)

    return NextResponse.json(
      {
        available: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 200 }, // Return 200 to the client but with error info
    )
  }
}
