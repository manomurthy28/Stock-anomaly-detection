import { AnomalyHistory } from "@/components/anomaly-history"

export default function AnomaliesPage() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Anomaly History</h1>

      <AnomalyHistory />
    </main>
  )
}
