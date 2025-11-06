import { NextResponse } from "next/server"
import { storeAnomalies } from "@/lib/anomaly-service"
import { logSystemEvent, logUserAction } from "@/lib/logger"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { stockSymbol, anomalies, stockPrices } = body

    if (!stockSymbol || !anomalies || !stockPrices) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    await logUserAction("analyze", `Analyzed stock: ${stockSymbol}`)

    const success = await storeAnomalies(stockSymbol, anomalies, stockPrices)

    if (success) {
      return NextResponse.json({ success: true })
    } else {
      return NextResponse.json({ error: "Failed to store anomalies" }, { status: 500 })
    }
  } catch (error) {
    console.error("Error in store-anomaly API:", error)
    await logSystemEvent("error", "API error in store-anomaly", String(error))
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
