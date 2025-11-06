"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Radar, Bar, Line } from "react-chartjs-2"
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
} from "chart.js"

// Register Chart.js components
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
)

interface ModelComparisonProps {
  data: {
    isolationForest: number[]
    lstm: number[]
  }
  features: string[]
}

export default function ModelComparison({ data, features }: ModelComparisonProps) {
  const chartData = {
    labels: features,
    datasets: [
      {
        label: "Isolation Forest",
        data: data.isolationForest,
        backgroundColor: "rgba(59, 130, 246, 0.2)",
        borderColor: "rgb(59, 130, 246)",
        borderWidth: 2,
        pointBackgroundColor: "rgb(59, 130, 246)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgb(59, 130, 246)",
      },
      {
        label: "LSTM Autoencoder",
        data: data.lstm,
        backgroundColor: "rgba(16, 185, 129, 0.2)",
        borderColor: "rgb(16, 185, 129)",
        borderWidth: 2,
        pointBackgroundColor: "rgb(16, 185, 129)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgb(16, 185, 129)",
      },
    ],
  }

  const options = {
    scales: {
      r: {
        beginAtZero: true,
        max: 1,
        ticks: {
          stepSize: 0.2,
        },
      },
    },
    elements: {
      line: {
        tension: 0.2,
      },
    },
  }

  // Performance metrics data
  const performanceData = {
    labels: ["Training Time", "Prediction Speed", "Memory Usage", "Scalability", "Complexity"],
    datasets: [
      {
        label: "Isolation Forest",
        data: [0.2, 0.15, 0.25, 0.7, 0.3],
        backgroundColor: "rgba(59, 130, 246, 0.7)",
      },
      {
        label: "LSTM Autoencoder",
        data: [0.8, 0.6, 0.7, 0.4, 0.8],
        backgroundColor: "rgba(16, 185, 129, 0.7)",
      },
    ],
  }

  const performanceOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: "Relative Resource Usage (lower is better)",
        },
      },
      x: {
        title: {
          display: true,
          text: "Performance Metrics",
        },
      },
    },
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: false,
        text: "Model Performance Comparison",
      },
    },
  }

  // Detection capability data
  const detectionData = {
    labels: ["Point Anomalies", "Contextual Anomalies", "Collective Anomalies", "Subtle Patterns", "Temporal Patterns"],
    datasets: [
      {
        label: "Isolation Forest",
        data: [0.9, 0.6, 0.4, 0.5, 0.3],
        backgroundColor: "rgba(59, 130, 246, 0.7)",
      },
      {
        label: "LSTM Autoencoder",
        data: [0.7, 0.8, 0.9, 0.85, 0.95],
        backgroundColor: "rgba(16, 185, 129, 0.7)",
      },
    ],
  }

  const detectionOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: "Detection Capability (higher is better)",
        },
      },
      x: {
        title: {
          display: true,
          text: "Anomaly Types",
        },
      },
    },
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: false,
        text: "Anomaly Detection Capability",
      },
    },
  }

  // Accuracy over data size
  const dataSizeLabels = ["100", "500", "1000", "5000", "10000"]
  const accuracyData = {
    labels: dataSizeLabels,
    datasets: [
      {
        label: "Isolation Forest",
        data: [0.82, 0.85, 0.87, 0.88, 0.89],
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.5)",
        tension: 0.3,
      },
      {
        label: "LSTM Autoencoder",
        data: [0.65, 0.75, 0.83, 0.91, 0.94],
        borderColor: "rgb(16, 185, 129)",
        backgroundColor: "rgba(16, 185, 129, 0.5)",
        tension: 0.3,
      },
    ],
  }

  const accuracyOptions = {
    responsive: true,
    scales: {
      y: {
        beginAtZero: false,
        min: 0.6,
        max: 1,
        title: {
          display: true,
          text: "Accuracy",
        },
      },
      x: {
        title: {
          display: true,
          text: "Training Data Size",
        },
      },
    },
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: false,
        text: "Accuracy vs Data Size",
      },
    },
  }

  return (
    <div className="grid grid-cols-1 gap-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Model Feature Importance</CardTitle>
            <CardDescription>Comparing how each model weighs different features</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex items-center justify-center">
              <Radar data={chartData} options={options} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>Computational resource requirements (lower is better)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex items-center justify-center">
              <Bar data={performanceData} options={performanceOptions} />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Anomaly Detection Capability</CardTitle>
            <CardDescription>Effectiveness at detecting different types of anomalies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex items-center justify-center">
              <Bar data={detectionData} options={detectionOptions} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Accuracy vs Data Size</CardTitle>
            <CardDescription>How model accuracy scales with training data size</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex items-center justify-center">
              <Line data={accuracyData} options={accuracyOptions} />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
