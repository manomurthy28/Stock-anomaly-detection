import prisma from "./prisma"

export type SystemEventType = "info" | "warning" | "error"
export type UserActionType = "search" | "view_anomaly" | "export" | "analyze"

export async function logSystemEvent(type: SystemEventType, message: string, details?: string) {
  try {
    await prisma.systemEvent.create({
      data: {
        type,
        message,
        details,
      },
    })
    console.log(`[${type.toUpperCase()}] ${message}`)
    return true
  } catch (error) {
    console.error("Failed to log system event:", error)
    return false
  }
}

export async function logUserAction(action: UserActionType, details?: string, userId?: string) {
  try {
    await prisma.userAction.create({
      data: {
        action,
        details,
        userId,
      },
    })
    return true
  } catch (error) {
    console.error("Failed to log user action:", error)
    return false
  }
}

export async function getRecentSystemEvents(limit = 50) {
  return prisma.systemEvent.findMany({
    take: limit,
    orderBy: {
      createdAt: "desc",
    },
  })
}

export async function getRecentUserActions(limit = 50) {
  return prisma.userAction.findMany({
    take: limit,
    orderBy: {
      createdAt: "desc",
    },
  })
}
