"use client"

import type React from "react"

import { useState } from "react"
import StockChart from "@/components/stock-chart"
import AnomalyMetrics from "@/components/anomaly-metrics"
import ModelComparison from "@/components/model-comparison"
import ModelAccuracy from "@/components/model-accuracy"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertCircle, BarChart3, LineChart, TrendingUp, CheckCircle, Loader2 } from "lucide-react"
import { fetchStockData } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"
import { NotificationBell } from "@/components/notification-bell"
import { WatchButton } from "@/components/watch-button"
import { Toaster } from "@/components/ui/toaster"
import { DatabaseStatus } from "@/components/database-status"
import { StockSymbolTester } from "@/components/stock-symbol-tester"
import { useToast } from "@/hooks/use-toast"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function Home() {
  const [stockSymbol, setStockSymbol] = useState("")
  const [validatedSymbol, setValidatedSymbol] = useState("")
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<any>(null)
  const [error, setError] = useState("")
  const [showTester, setShowTester] = useState(false)
  const { toast } = useToast()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!stockSymbol) return

    // For Indian stocks, show the tester first
    if (stockSymbol.endsWith(".NS") || stockSymbol.endsWith(".BO")) {
      setShowTester(true)
      return
    }

    await analyzeStock(stockSymbol)
  }

  const analyzeStock = async (symbol: string) => {
    setLoading(true)
    setError("")
    setShowTester(false)

    try {
      toast({
        title: "Analyzing Stock",
        description: `Starting analysis for ${symbol}. This may take a moment...`,
        variant: "default",
      })

      const result = await fetchStockData(symbol)
      setData(result)

      toast({
        title: "Analysis Complete",
        description: `Successfully analyzed ${symbol}`,
        variant: "default",
      })
    } catch (err: any) {
      const errorMessage = err.message || "Unknown error"
      setError(`Failed to fetch stock data: ${errorMessage}`)
      console.error(err)

      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const handleSymbolValidated = (symbol: string) => {
    if (!symbol) {
      // User cancelled
      setShowTester(false)
      return
    }

    setValidatedSymbol(symbol)
    analyzeStock(symbol)
  }

  return (
    <main className="min-h-screen bg-gray-50 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col gap-6">
          <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Stock Anomaly Detection</h1>
              <p className="text-gray-500 mt-1">
                Compare Isolation Forest and LSTM Autoencoder models for detecting market anomalies
              </p>
            </div>

            <div className="flex items-center gap-2">
              <NotificationBell />
              <form onSubmit={handleSubmit} className="flex w-full md:w-auto gap-2">
                <Input
                  type="text"
                  placeholder="Enter stock symbol (e.g., AAPL, SBIN.NS)"
                  value={stockSymbol}
                  onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
                  className="w-full md:w-64"
                />
                <Button type="submit" disabled={loading || !stockSymbol}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    "Analyze"
                  )}
                </Button>
              </form>
            </div>
          </header>

          <DatabaseStatus />

          {showTester && <StockSymbolTester onSymbolValidated={handleSymbolValidated} />}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error analyzing stock</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <Skeleton className="h-8 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-[300px] w-full" />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <Skeleton className="h-8 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-[300px] w-full" />
                </CardContent>
              </Card>
            </div>
          ) : data ? (
            <>
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">{data.stockSymbol} Analysis</h2>
                <div className="flex items-center gap-2">
                  {data.dataSource && <span className="text-sm text-gray-500">Source: {data.dataSource}</span>}
                  <WatchButton stockSymbol={data.stockSymbol} />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium">Stock Price</CardTitle>
                    <CardDescription>Last 30 days performance</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[120px]">
                      <StockChart data={data.stockPrices.slice(-30)} type="line" showAnomalies={false} />
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium">Anomaly Count</CardTitle>
                    <CardDescription>Detected anomalies by model</CardDescription>
                  </CardHeader>
                  <CardContent className="flex items-center justify-center h-[120px]">
                    <div className="grid grid-cols-2 w-full gap-4">
                      <div className="flex flex-col items-center">
                        <span className="text-3xl font-bold">{data.anomalyCount.isolationForest}</span>
                        <span className="text-sm text-gray-500">Isolation Forest</span>
                      </div>
                      <div className="flex flex-col items-center">
                        <span className="text-3xl font-bold">{data.anomalyCount.lstm}</span>
                        <span className="text-sm text-gray-500">LSTM Autoencoder</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium">Model Agreement</CardTitle>
                    <CardDescription>Anomaly detection overlap</CardDescription>
                  </CardHeader>
                  <CardContent className="flex items-center justify-center h-[120px]">
                    <div className="text-center">
                      <div className="text-3xl font-bold">{data.modelAgreement.percentage}%</div>
                      <div className="text-sm text-gray-500">{data.modelAgreement.count} shared anomalies</div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Tabs defaultValue="visualization">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="visualization" className="flex items-center gap-2">
                    <LineChart size={16} />
                    <span>Visualization</span>
                  </TabsTrigger>
                  <TabsTrigger value="comparison" className="flex items-center gap-2">
                    <BarChart3 size={16} />
                    <span>Model Comparison</span>
                  </TabsTrigger>
                  <TabsTrigger value="metrics" className="flex items-center gap-2">
                    <TrendingUp size={16} />
                    <span>Performance Metrics</span>
                  </TabsTrigger>
                  <TabsTrigger value="accuracy" className="flex items-center gap-2">
                    <CheckCircle size={16} />
                    <span>Model Accuracy</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="visualization" className="mt-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Stock Price with Detected Anomalies</CardTitle>
                      <CardDescription>
                        Visualizing anomalies detected by both models on the price chart
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[400px]">
                        <StockChart
                          data={data.stockPrices}
                          anomaliesIF={data.anomalies.isolationForest}
                          anomaliesLSTM={data.anomalies.lstm}
                          type="line"
                          showAnomalies={true}
                        />
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="comparison" className="mt-6">
                  <ModelComparison data={data.modelComparison} features={data.features} />
                </TabsContent>

                <TabsContent value="metrics" className="mt-6">
                  <AnomalyMetrics
                    isolationForestMetrics={data.metrics.isolationForest}
                    lstmMetrics={data.metrics.lstm}
                  />
                </TabsContent>

                <TabsContent value="accuracy" className="mt-6">
                  {data.validation ? (
                    <ModelAccuracy
                      isolationForestMetrics={data.validation.isolationForest}
                      lstmMetrics={data.validation.lstm}
                    />
                  ) : (
                    <Card>
                      <CardHeader>
                        <CardTitle>Model Accuracy</CardTitle>
                        <CardDescription>
                          Validation metrics based on market movements following anomaly detection
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-col items-center justify-center py-8 text-center">
                          <div className="bg-gray-100 p-4 rounded-full mb-4">
                            <AlertCircle size={32} className="text-gray-400" />
                          </div>
                          <h3 className="text-lg font-medium mb-2">Accuracy data not available</h3>
                          <p className="text-gray-500 max-w-md">
                            The backend needs to be updated to include model validation metrics.
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="bg-gray-100 p-6 rounded-full mb-4">
                <TrendingUp size={48} className="text-gray-400" />
              </div>
              <h2 className="text-xl font-medium mb-2">Enter a stock symbol to begin analysis</h2>
              <p className="text-gray-500 max-w-md">
                The dashboard will compare anomaly detection results from Isolation Forest and LSTM Autoencoder models.
              </p>
              <div className="mt-4 text-sm text-gray-500">
                <p>For Indian stocks, use the .NS suffix for NSE (e.g., SBIN.NS, TCS.NS)</p>
                <p>For BSE stocks, use the .BO suffix (e.g., RELIANCE.BO)</p>
              </div>
            </div>
          )}
        </div>
      </div>
      <Toaster />
    </main>
  )
}
