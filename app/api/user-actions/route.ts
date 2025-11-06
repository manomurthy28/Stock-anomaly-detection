import { NextResponse } from "next/server"
import { getRecentUserActions } from "@/lib/logger"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "50")

    const actions = await getRecentUserActions(limit)

    return NextResponse.json({ actions })
  } catch (error) {
    console.error("Error in user-actions API:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
