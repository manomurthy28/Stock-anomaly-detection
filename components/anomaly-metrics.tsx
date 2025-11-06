"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface MetricsProps {
  precision: number
  recall: number
  f1Score: number
  rocAuc: number
  executionTime: number
}

interface AnomalyMetricsProps {
  isolationForestMetrics: MetricsProps
  lstmMetrics: MetricsProps
}

export default function AnomalyMetrics({ isolationForestMetrics, lstmMetrics }: AnomalyMetricsProps) {
  return (
    <Tabs defaultValue="isolation-forest">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="isolation-forest">Isolation Forest</TabsTrigger>
        <TabsTrigger value="lstm">LSTM Autoencoder</TabsTrigger>
      </TabsList>

      <TabsContent value="isolation-forest" className="mt-4">
        <Card>
          <CardHeader>
            <CardTitle>Isolation Forest Performance Metrics</CardTitle>
            <CardDescription>Evaluation metrics for the Isolation Forest anomaly detection model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MetricCard
                title="Precision"
                value={isolationForestMetrics.precision}
                description="Ratio of correctly identified anomalies to all predicted anomalies"
              />
              <MetricCard
                title="Recall"
                value={isolationForestMetrics.recall}
                description="Ratio of correctly identified anomalies to all actual anomalies"
              />
              <MetricCard
                title="F1 Score"
                value={isolationForestMetrics.f1Score}
                description="Harmonic mean of precision and recall"
              />
              <MetricCard
                title="ROC AUC"
                value={isolationForestMetrics.rocAuc}
                description="Area under the ROC curve"
              />
            </div>

            <div className="mt-6">
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">Execution Time</span>
                <span className="text-sm font-medium">{isolationForestMetrics.executionTime.toFixed(2)}s</span>
              </div>
              <Progress value={isolationForestMetrics.executionTime * 10} className="h-2" />
              <p className="text-xs text-gray-500 mt-1">Time taken to train the model and detect anomalies</p>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="lstm" className="mt-4">
        <Card>
          <CardHeader>
            <CardTitle>LSTM Autoencoder Performance Metrics</CardTitle>
            <CardDescription>Evaluation metrics for the LSTM Autoencoder anomaly detection model</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <MetricCard
                title="Precision"
                value={lstmMetrics.precision}
                description="Ratio of correctly identified anomalies to all predicted anomalies"
              />
              <MetricCard
                title="Recall"
                value={lstmMetrics.recall}
                description="Ratio of correctly identified anomalies to all actual anomalies"
              />
              <MetricCard
                title="F1 Score"
                value={lstmMetrics.f1Score}
                description="Harmonic mean of precision and recall"
              />
              <MetricCard title="ROC AUC" value={lstmMetrics.rocAuc} description="Area under the ROC curve" />
            </div>

            <div className="mt-6">
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">Execution Time</span>
                <span className="text-sm font-medium">{lstmMetrics.executionTime.toFixed(2)}s</span>
              </div>
              <Progress value={lstmMetrics.executionTime * 5} className="h-2" />
              <p className="text-xs text-gray-500 mt-1">Time taken to train the model and detect anomalies</p>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}

interface MetricCardProps {
  title: string
  value: number
  description: string
}

function MetricCard({ title, value, description }: MetricCardProps) {
  // Convert value to percentage for display
  const percentage = (value * 100).toFixed(2)

  return (
    <div className="bg-gray-50 p-4 rounded-lg">
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-sm font-medium">{title}</h3>
        <span className="text-lg font-bold">{percentage}%</span>
      </div>
      <Progress value={value * 100} className="h-2 mb-2" />
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  )
}
