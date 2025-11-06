"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Bar } from "react-chartjs-2"
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js"

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

interface AccuracyMetrics {
  truePositives: number
  falsePositives: number
  trueNegatives: number
  falseNegatives: number
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  historicalAccuracy: {
    dates: string[]
    values: number[]
  }
}

interface ModelAccuracyProps {
  isolationForestMetrics: AccuracyMetrics
  lstmMetrics: AccuracyMetrics
}

export default function ModelAccuracy({ isolationForestMetrics, lstmMetrics }: ModelAccuracyProps) {
  // Ensure we have all required data
  if (!isolationForestMetrics || !lstmMetrics) {
    console.error("Missing metrics data for ModelAccuracy component")
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Accuracy</CardTitle>
          <CardDescription>Data not available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p>Accuracy metrics data is not available.</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Prepare confusion matrix data for Isolation Forest
  const ifConfusionData = {
    labels: ["True Positives", "False Positives", "True Negatives", "False Negatives"],
    datasets: [
      {
        label: "Count",
        data: [
          isolationForestMetrics.truePositives,
          isolationForestMetrics.falsePositives,
          isolationForestMetrics.trueNegatives,
          isolationForestMetrics.falseNegatives,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.6)", // True Positives - Green
          "rgba(255, 99, 132, 0.6)", // False Positives - Red
          "rgba(54, 162, 235, 0.6)", // True Negatives - Blue
          "rgba(255, 206, 86, 0.6)", // False Negatives - Yellow
        ],
        borderColor: [
          "rgba(75, 192, 192, 1)",
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
        ],
        borderWidth: 1,
      },
    ],
  }

  // Prepare confusion matrix data for LSTM
  const lstmConfusionData = {
    labels: ["True Positives", "False Positives", "True Negatives", "False Negatives"],
    datasets: [
      {
        label: "Count",
        data: [
          lstmMetrics.truePositives,
          lstmMetrics.falsePositives,
          lstmMetrics.trueNegatives,
          lstmMetrics.falseNegatives,
        ],
        backgroundColor: [
          "rgba(75, 192, 192, 0.6)",
          "rgba(255, 99, 132, 0.6)",
          "rgba(54, 162, 235, 0.6)",
          "rgba(255, 206, 86, 0.6)",
        ],
        borderColor: [
          "rgba(75, 192, 192, 1)",
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
        ],
        borderWidth: 1,
      },
    ],
  }

  // Ensure we have historical accuracy data
  const ifHistoricalDates = isolationForestMetrics.historicalAccuracy?.dates || []
  const ifHistoricalValues = isolationForestMetrics.historicalAccuracy?.values || []
  const lstmHistoricalDates = lstmMetrics.historicalAccuracy?.dates || []
  const lstmHistoricalValues = lstmMetrics.historicalAccuracy?.values || []

  // Prepare historical accuracy data for Isolation Forest
  const ifHistoricalData = {
    labels: ifHistoricalDates,
    datasets: [
      {
        label: "Accuracy",
        data: ifHistoricalValues,
        backgroundColor: "rgba(59, 130, 246, 0.6)",
        borderColor: "rgb(59, 130, 246)",
        borderWidth: 1,
      },
    ],
  }

  // Prepare historical accuracy data for LSTM
  const lstmHistoricalData = {
    labels: lstmHistoricalDates,
    datasets: [
      {
        label: "Accuracy",
        data: lstmHistoricalValues,
        backgroundColor: "rgba(16, 185, 129, 0.6)",
        borderColor: "rgb(16, 185, 129)",
        borderWidth: 1,
      },
    ],
  }

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  }

  return (
    <Tabs defaultValue="isolation-forest">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="isolation-forest">Isolation Forest</TabsTrigger>
        <TabsTrigger value="lstm">LSTM Autoencoder</TabsTrigger>
      </TabsList>

      <TabsContent value="isolation-forest" className="mt-4 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Isolation Forest Accuracy Metrics</CardTitle>
            <CardDescription>Validation metrics based on market movements following anomaly detection</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Accuracy</span>
                    <span className="text-sm font-medium">{(isolationForestMetrics.accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={isolationForestMetrics.accuracy * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">
                    Percentage of correctly identified anomalies and non-anomalies
                  </p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Precision</span>
                    <span className="text-sm font-medium">{(isolationForestMetrics.precision * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={isolationForestMetrics.precision * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">
                    Percentage of detected anomalies that were actual market events
                  </p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Recall</span>
                    <span className="text-sm font-medium">{(isolationForestMetrics.recall * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={isolationForestMetrics.recall * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">Percentage of actual market events that were detected</p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">F1 Score</span>
                    <span className="text-sm font-medium">{(isolationForestMetrics.f1Score * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={isolationForestMetrics.f1Score * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">Harmonic mean of precision and recall</p>
                </div>
              </div>

              <div className="h-[300px]">
                <h3 className="text-sm font-medium mb-2">Confusion Matrix</h3>
                <Bar data={ifConfusionData} options={options} />
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium mb-2">Historical Accuracy</h3>
              <div className="h-[200px]">
                <Bar data={ifHistoricalData} options={options} />
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="lstm" className="mt-4 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>LSTM Autoencoder Accuracy Metrics</CardTitle>
            <CardDescription>Validation metrics based on market movements following anomaly detection</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Accuracy</span>
                    <span className="text-sm font-medium">{(lstmMetrics.accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={lstmMetrics.accuracy * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">
                    Percentage of correctly identified anomalies and non-anomalies
                  </p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Precision</span>
                    <span className="text-sm font-medium">{(lstmMetrics.precision * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={lstmMetrics.precision * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">
                    Percentage of detected anomalies that were actual market events
                  </p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Recall</span>
                    <span className="text-sm font-medium">{(lstmMetrics.recall * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={lstmMetrics.recall * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">Percentage of actual market events that were detected</p>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">F1 Score</span>
                    <span className="text-sm font-medium">{(lstmMetrics.f1Score * 100).toFixed(2)}%</span>
                  </div>
                  <Progress value={lstmMetrics.f1Score * 100} className="h-2" />
                  <p className="text-xs text-gray-500 mt-1">Harmonic mean of precision and recall</p>
                </div>
              </div>

              <div className="h-[300px]">
                <h3 className="text-sm font-medium mb-2">Confusion Matrix</h3>
                <Bar data={lstmConfusionData} options={options} />
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium mb-2">Historical Accuracy</h3>
              <div className="h-[200px]">
                <Bar data={lstmHistoricalData} options={options} />
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}
