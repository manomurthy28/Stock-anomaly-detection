"use client"

import { useRef } from "react"
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  type ChartData,
  type ChartOptions,
} from "chart.js"
import { Line, Bar } from "react-chartjs-2"

// Register Chart.js components
Chart.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend)

interface StockChartProps {
  data: Array<{
    date: string
    price: number
  }>
  anomaliesIF?: number[]
  anomaliesLSTM?: number[]
  type: "line" | "bar"
  showAnomalies: boolean
}

export default function StockChart({
  data,
  anomaliesIF = [],
  anomaliesLSTM = [],
  type = "line",
  showAnomalies = true,
}: StockChartProps) {
  const chartRef = useRef<Chart | null>(null)

  const labels = data.map((item) => item.date)
  const prices = data.map((item) => item.price)

  const chartData: ChartData<"line" | "bar"> = {
    labels,
    datasets: [
      {
        label: "Stock Price",
        data: prices,
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 2,
        tension: 0.1,
        fill: true,
      },
    ],
  }

  if (showAnomalies) {
    // Add LSTM Autoencoder anomalies FIRST (so they appear below Isolation Forest)
    // This helps prevent them from being hidden
    chartData.datasets.push({
      label: "LSTM Autoencoder Anomalies",
      data: data.map((_, index) => (anomaliesLSTM.includes(index) ? prices[index] : null)),
      borderColor: "rgba(16, 185, 129, 1)", // Increased opacity
      backgroundColor: "rgba(16, 185, 129, 1)", // Increased opacity
      borderWidth: 2, // Added border width
      pointRadius: 8, // Increased point size
      pointStyle: "triangle", // Distinctive shape
      pointHoverRadius: 10, // Increased hover size
      type: "scatter",
      // Offset the points slightly to prevent overlap
      pointHitRadius: 10, // Larger hit area for better interaction
    })

    // Add Isolation Forest anomalies SECOND
    chartData.datasets.push({
      label: "Isolation Forest Anomalies",
      data: data.map((_, index) => (anomaliesIF.includes(index) ? prices[index] : null)),
      borderColor: "rgba(239, 68, 68, 1)", // Increased opacity
      backgroundColor: "rgba(239, 68, 68, 1)", // Increased opacity
      borderWidth: 2, // Added border width
      pointRadius: 6, // Slightly smaller than LSTM
      pointStyle: "circle", // Different shape from LSTM
      pointHoverRadius: 8, // Increased hover size
      type: "scatter",
      pointHitRadius: 10, // Larger hit area for better interaction
    })
  }

  const options: ChartOptions<"line" | "bar"> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
        },
      },
      tooltip: {
        enabled: true,
      },
    },
    scales: {
      x: {
        ticks: {
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 10,
        },
        grid: {
          display: false,
        },
      },
      y: {
        beginAtZero: false,
        grid: {
          drawBorder: false,
        },
      },
    },
  }

  return (
    <div className="w-full h-full">
      {type === "line" ? <Line data={chartData} options={options} /> : <Bar data={chartData} options={options} />}
    </div>
  )
}
