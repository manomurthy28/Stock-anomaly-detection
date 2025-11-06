"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, CheckCircle, XCircle } from "lucide-react"

export function DatabaseStatus() {
  const [status, setStatus] = useState<"loading" | "connected" | "error">("loading")
  const [message, setMessage] = useState("Checking database connection...")

  useEffect(() => {
    const checkDatabaseStatus = async () => {
      try {
        const response = await fetch("/api/db-status")
        const data = await response.json()

        if (data.connected) {
          setStatus("connected")
          setMessage(`Connected to PostgreSQL database. ${data.tables || 0} tables available.`)
        } else {
          setStatus("error")
          setMessage(data.error || "Failed to connect to database")
        }
      } catch (error) {
        setStatus("error")
        setMessage("Failed to check database status")
      }
    }

    checkDatabaseStatus()
  }, [])

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-medium flex items-center">
          <Database className="mr-2 h-5 w-5" />
          Database Status
        </CardTitle>
        <CardDescription>PostgreSQL connection status</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center">
          {status === "loading" ? (
            <Badge className="bg-yellow-100 text-yellow-800">Connecting...</Badge>
          ) : status === "connected" ? (
            <Badge className="bg-green-100 text-green-800 flex items-center">
              <CheckCircle className="mr-1 h-3 w-3" />
              Connected
            </Badge>
          ) : (
            <Badge className="bg-red-100 text-red-800 flex items-center">
              <XCircle className="mr-1 h-3 w-3" />
              Error
            </Badge>
          )}
          <span className="ml-2 text-sm text-gray-600">{message}</span>
        </div>
      </CardContent>
    </Card>
  )
}
