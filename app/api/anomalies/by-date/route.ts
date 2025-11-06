import { NextResponse } from "next/server"
import { getAnomaliesByDate } from "@/lib/anomaly-service"
import { logSystemEvent, logUserAction } from "@/lib/logger"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)

    const startDateParam = searchParams.get("startDate")
    const endDateParam = searchParams.get("endDate")
    const model = searchParams.get("model") || undefined
    const stockSymbol = searchParams.get("stockSymbol") || undefined

    if (!startDateParam || !endDateParam) {
      return NextResponse.json({ error: "Start date and end date are required" }, { status: 400 })
    }

    const startDate = new Date(startDateParam)
    const endDate = new Date(endDateParam)

    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return NextResponse.json({ error: "Invalid date format" }, { status: 400 })
    }

    await logUserAction(
      "view_anomaly",
      `Viewed anomalies from ${startDateParam} to ${endDateParam}${
        model ? ` for model ${model}` : ""
      }${stockSymbol ? ` for stock ${stockSymbol}` : ""}`,
    )

    const anomalies = await getAnomaliesByDate(startDate, endDate, model, stockSymbol)

    return NextResponse.json({ anomalies })
  } catch (error) {
    console.error("Error in anomalies/by-date API:", error)
    await logSystemEvent("error", "API error in anomalies/by-date", String(error))
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
