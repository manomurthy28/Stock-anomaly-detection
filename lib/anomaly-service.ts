import prisma from "./prisma"
import { logSystemEvent } from "./logger"

export async function storeAnomalies(
  stockSymbol: string,
  anomalies: {
    isolationForest: number[]
    lstm: number[]
  },
  stockPrices: { date: string; price: number }[],
) {
  try {
    // Find or create the stock
    let stock = await prisma.stock.findUnique({
      where: { symbol: stockSymbol },
    })

    if (!stock) {
      stock = await prisma.stock.create({
        data: { symbol: stockSymbol },
      })
      await logSystemEvent("info", `Created new stock: ${stockSymbol}`)
    }

    // Process isolation forest anomalies
    for (const index of anomalies.isolationForest) {
      if (index >= 0 && index < stockPrices.length) {
        const isShared = anomalies.lstm.includes(index)
        const detectedBy = isShared ? "both" : "isolationForest"

        await prisma.anomaly.create({
          data: {
            stockId: stock.id,
            date: new Date(stockPrices[index].date),
            index,
            price: stockPrices[index].price,
            detectedBy,
            confidence: 0.85, // Mock confidence value
          },
        })
      }
    }

    // Process LSTM anomalies (only those not already processed as 'both')
    for (const index of anomalies.lstm) {
      if (index >= 0 && index < stockPrices.length && !anomalies.isolationForest.includes(index)) {
        await prisma.anomaly.create({
          data: {
            stockId: stock.id,
            date: new Date(stockPrices[index].date),
            index,
            price: stockPrices[index].price,
            detectedBy: "lstm",
            confidence: 0.82, // Mock confidence value
          },
        })
      }
    }

    await logSystemEvent(
      "info",
      `Stored anomalies for ${stockSymbol}`,
      `IF: ${anomalies.isolationForest.length}, LSTM: ${anomalies.lstm.length}`,
    )

    return true
  } catch (error) {
    console.error("Failed to store anomalies:", error)
    await logSystemEvent("error", `Failed to store anomalies for ${stockSymbol}`, String(error))
    return false
  }
}

export async function getAnomaliesByStock(stockSymbol: string) {
  try {
    const stock = await prisma.stock.findUnique({
      where: { symbol: stockSymbol },
    })

    if (!stock) {
      return []
    }

    return prisma.anomaly.findMany({
      where: { stockId: stock.id },
      orderBy: { date: "desc" },
    })
  } catch (error) {
    console.error("Failed to get anomalies:", error)
    await logSystemEvent("error", `Failed to get anomalies for ${stockSymbol}`, String(error))
    return []
  }
}

export async function getAnomaliesByDate(startDate: Date, endDate: Date, model?: string, stockSymbol?: string) {
  try {
    const where: any = {
      date: {
        gte: startDate,
        lte: endDate,
      },
    }

    if (model) {
      where.detectedBy = model
    }

    if (stockSymbol) {
      const stock = await prisma.stock.findUnique({
        where: { symbol: stockSymbol },
      })
      if (stock) {
        where.stockId = stock.id
      } else {
        return []
      }
    }

    return prisma.anomaly.findMany({
      where,
      include: {
        stock: true,
      },
      orderBy: { date: "desc" },
    })
  } catch (error) {
    console.error("Failed to get anomalies by date:", error)
    await logSystemEvent("error", "Failed to get anomalies by date", String(error))
    return []
  }
}
