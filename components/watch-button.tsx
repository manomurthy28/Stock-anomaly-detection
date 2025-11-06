"use client"

import { Eye, EyeOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useNotifications } from "@/contexts/notification-context"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface WatchButtonProps {
  stockSymbol: string
}

export function WatchButton({ stockSymbol }: WatchButtonProps) {
  const { isWatching, watchStock, unwatchStock } = useNotifications()
  const watching = isWatching(stockSymbol)

  const handleClick = () => {
    if (watching) {
      unwatchStock(stockSymbol)
    } else {
      watchStock(stockSymbol)
    }
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant={watching ? "default" : "outline"} size="sm" onClick={handleClick} className="gap-1">
            {watching ? (
              <>
                <EyeOff className="h-4 w-4" />
                <span>Unwatch</span>
              </>
            ) : (
              <>
                <Eye className="h-4 w-4" />
                <span>Watch</span>
              </>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          {watching ? "Stop receiving anomaly alerts for this stock" : "Get notified when new anomalies are detected"}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
