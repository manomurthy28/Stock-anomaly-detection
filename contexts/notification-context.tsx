"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import { fetchStockData } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"

type Notification = {
  id: string
  title: string
  message: string
  type: "info" | "warning" | "success" | "error"
  timestamp: Date
  read: boolean
  stockSymbol?: string
}

type WatchedStock = {
  symbol: string
  lastChecked: Date
  knownAnomalies: {
    isolationForest: number[]
    lstm: number[]
  }
}

type NotificationContextType = {
  notifications: Notification[]
  watchedStocks: WatchedStock[]
  addNotification: (notification: Omit<Notification, "id" | "timestamp" | "read">) => void
  markAsRead: (id: string) => void
  clearNotifications: () => void
  watchStock: (symbol: string) => void
  unwatchStock: (symbol: string) => void
  isWatching: (symbol: string) => boolean
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined)

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [watchedStocks, setWatchedStocks] = useState<WatchedStock[]>([])
  const { toast } = useToast()

  // Load watched stocks from localStorage on mount
  useEffect(() => {
    const savedStocks = localStorage.getItem("watchedStocks")
    if (savedStocks) {
      try {
        const parsed = JSON.parse(savedStocks)
        // Convert string dates back to Date objects
        const stocks = parsed.map((stock: any) => ({
          ...stock,
          lastChecked: new Date(stock.lastChecked),
        }))
        setWatchedStocks(stocks)
      } catch (e) {
        console.error("Failed to parse watched stocks from localStorage", e)
      }
    }
  }, [])

  // Save watched stocks to localStorage when they change
  useEffect(() => {
    localStorage.setItem("watchedStocks", JSON.stringify(watchedStocks))
  }, [watchedStocks])

  // Check for new anomalies every 5 minutes
  useEffect(() => {
    if (watchedStocks.length === 0) return

    const checkForAnomalies = async () => {
      for (const stock of watchedStocks) {
        try {
          // Only check stocks that haven't been checked in the last 5 minutes
          const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000)
          if (stock.lastChecked > fiveMinutesAgo) continue

          const data = await fetchStockData(stock.symbol)

          // Find new anomalies (ones that weren't in the known anomalies list)
          const newIFAnomalies = data.anomalies.isolationForest.filter(
            (a: number) => !stock.knownAnomalies.isolationForest.includes(a),
          )

          const newLSTMAnomalies = data.anomalies.lstm.filter((a: number) => !stock.knownAnomalies.lstm.includes(a))

          // Update the last checked time and known anomalies
          setWatchedStocks((prev) =>
            prev.map((s) =>
              s.symbol === stock.symbol
                ? {
                    ...s,
                    lastChecked: new Date(),
                    knownAnomalies: data.anomalies,
                  }
                : s,
            ),
          )

          // If there are new anomalies, create notifications
          if (newIFAnomalies.length > 0) {
            const notification = {
              title: `New Anomalies Detected (IF)`,
              message: `Isolation Forest detected ${newIFAnomalies.length} new anomalies in ${stock.symbol}.`,
              type: "warning" as const,
              stockSymbol: stock.symbol,
            }

            addNotification(notification)

            // Also show a toast notification
            toast({
              title: notification.title,
              description: notification.message,
              variant: "warning",
            })
          }

          if (newLSTMAnomalies.length > 0) {
            const notification = {
              title: `New Anomalies Detected (LSTM)`,
              message: `LSTM Autoencoder detected ${newLSTMAnomalies.length} new anomalies in ${stock.symbol}.`,
              type: "warning" as const,
              stockSymbol: stock.symbol,
            }

            addNotification(notification)

            // Also show a toast notification
            toast({
              title: notification.title,
              description: notification.message,
              variant: "warning",
            })
          }
        } catch (error) {
          console.error(`Error checking anomalies for ${stock.symbol}:`, error)
        }
      }
    }

    // Initial check
    checkForAnomalies()

    // Set up interval for periodic checks
    const intervalId = setInterval(checkForAnomalies, 5 * 60 * 1000) // Check every 5 minutes

    return () => clearInterval(intervalId)
  }, [watchedStocks, toast])

  const addNotification = (notification: Omit<Notification, "id" | "timestamp" | "read">) => {
    const newNotification: Notification = {
      ...notification,
      id: Date.now().toString(),
      timestamp: new Date(),
      read: false,
    }
    setNotifications((prev) => [newNotification, ...prev])
  }

  const markAsRead = (id: string) => {
    setNotifications((prev) =>
      prev.map((notification) => (notification.id === id ? { ...notification, read: true } : notification)),
    )
  }

  const clearNotifications = () => {
    setNotifications([])
  }

  const watchStock = (symbol: string) => {
    // Don't add if already watching
    if (watchedStocks.some((stock) => stock.symbol === symbol)) return

    setWatchedStocks((prev) => [
      ...prev,
      {
        symbol,
        lastChecked: new Date(0), // Set to epoch to ensure it gets checked immediately
        knownAnomalies: { isolationForest: [], lstm: [] },
      },
    ])

    const notification = {
      title: "Stock Added to Watchlist",
      message: `${symbol} has been added to your watchlist. You'll be notified of new anomalies.`,
      type: "success" as const,
    }

    addNotification(notification)

    // Also show a toast notification
    toast({
      title: notification.title,
      description: notification.message,
      variant: "success",
    })
  }

  const unwatchStock = (symbol: string) => {
    setWatchedStocks((prev) => prev.filter((stock) => stock.symbol !== symbol))

    const notification = {
      title: "Stock Removed from Watchlist",
      message: `${symbol} has been removed from your watchlist.`,
      type: "info" as const,
    }

    addNotification(notification)

    // Also show a toast notification
    toast({
      title: notification.title,
      description: notification.message,
    })
  }

  const isWatching = (symbol: string) => {
    return watchedStocks.some((stock) => stock.symbol === symbol)
  }

  return (
    <NotificationContext.Provider
      value={{
        notifications,
        watchedStocks,
        addNotification,
        markAsRead,
        clearNotifications,
        watchStock,
        unwatchStock,
        isWatching,
      }}
    >
      {children}
    </NotificationContext.Provider>
  )
}

export function useNotifications() {
  const context = useContext(NotificationContext)
  if (context === undefined) {
    throw new Error("useNotifications must be used within a NotificationProvider")
  }
  return context
}
