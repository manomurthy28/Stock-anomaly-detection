import { NextResponse } from "next/server"
import prisma from "@/lib/prisma"
import { logSystemEvent } from "@/lib/logger"

export async function GET() {
  try {
    // Try to query the database
    const tableNames = await prisma.$queryRaw`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public'
    `

    const tables = Array.isArray(tableNames) ? tableNames.length : 0

    return NextResponse.json({
      connected: true,
      tables,
      message: "Database connection successful",
    })
  } catch (error) {
    console.error("Database connection error:", error)

    // Log the error
    await logSystemEvent("error", "Database connection failed", String(error)).catch(console.error) // Handle error in logging

    return NextResponse.json({
      connected: false,
      error: String(error),
      message: "Failed to connect to database",
    })
  }
}
