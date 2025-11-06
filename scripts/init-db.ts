import { PrismaClient } from "@prisma/client"
import { logSystemEvent } from "../lib/logger"

const prisma = new PrismaClient()

async function main() {
  console.log("Starting database initialization...")

  try {
    // Log the initialization
    await logSystemEvent("info", "Database initialization started")

    // Create some sample system events
    await prisma.systemEvent.createMany({
      data: [
        {
          type: "info",
          message: "Application started",
          details: "The application was started successfully",
        },
        {
          type: "info",
          message: "Database connected",
          details: "Successfully connected to PostgreSQL database",
        },
      ],
      skipDuplicates: true,
    })

    console.log("Database initialization completed successfully")
    await logSystemEvent("info", "Database initialization completed")

    return { success: true }
  } catch (error) {
    console.error("Database initialization failed:", error)
    return { success: false, error }
  } finally {
    await prisma.$disconnect()
  }
}

// Execute if this file is run directly
if (require.main === module) {
  main()
    .then((result) => {
      if (result.success) {
        console.log("✅ Database initialization successful")
        process.exit(0)
      } else {
        console.error("❌ Database initialization failed:", result.error)
        process.exit(1)
      }
    })
    .catch((error) => {
      console.error("❌ Unhandled error during database initialization:", error)
      process.exit(1)
    })
}

export default main
