"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { format } from "date-fns"

export function EventLogs() {
  const [events, setEvents] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchEvents = async () => {
      setLoading(true)
      try {
        const response = await fetch("/api/system-events")
        const data = await response.json()
        setEvents(data.events)
      } catch (error) {
        console.error("Failed to fetch system events:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchEvents()
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Events</CardTitle>
        <CardDescription>Recent system events and logs</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="text-center py-4">Loading events...</div>
        ) : events.length > 0 ? (
          <div className="space-y-4">
            {events.map((event) => (
              <div key={event.id} className="border rounded-md p-4 bg-gray-50">
                <div className="flex justify-between items-start">
                  <div className="flex items-center">
                    <Badge
                      className={
                        event.type === "error"
                          ? "bg-red-100 text-red-800"
                          : event.type === "warning"
                            ? "bg-yellow-100 text-yellow-800"
                            : "bg-green-100 text-green-800"
                      }
                    >
                      {event.type}
                    </Badge>
                    <span className="ml-2 font-medium">{event.message}</span>
                  </div>
                  <span className="text-xs text-gray-500">{format(new Date(event.createdAt), "PPp")}</span>
                </div>
                {event.details && <div className="mt-2 text-sm text-gray-600 whitespace-pre-wrap">{event.details}</div>}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-4 text-gray-500">No system events found.</div>
        )}
      </CardContent>
    </Card>
  )
}
