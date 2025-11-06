import { EventLogs } from "@/components/event-logs"
import { UserActions } from "@/components/user-actions"

export default function LogsPage() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">System Logs & User Activity</h1>

      <div className="grid grid-cols-1 gap-6">
        <EventLogs />
        <UserActions />
      </div>
    </main>
  )
}
