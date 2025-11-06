import { NextResponse } from "next/server"
import { getRecentSystemEvents } from "@/lib/logger"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "50")

    const events = await getRecentSystemEvents(limit)

    return NextResponse.json({ events })
  } catch (error) {
    console.error("Error in system-events API:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
