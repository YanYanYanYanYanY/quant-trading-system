export type EngineMode = "BACKTEST" | "PAPER" | "LIVE";
export type RunState = "STOPPED" | "RUNNING" | "PAUSED" | "KILLED";

export type OrderStatus =
  | "NEW"
  | "SUBMITTED"
  | "ACCEPTED"
  | "PARTIALLY_FILLED"
  | "FILLED"
  | "CANCELED"
  | "REJECTED";

export interface AccountSnapshot {
  equity: number;
  cash: number;
  buyingPower: number;
  dailyPnl: number;
  updatedAt: string; // ISO
}

export interface Position {
  symbol: string;
  qty: number;
  avgPrice: number;
  marketPrice: number;
  unrealizedPnl: number;
  strategyTag?: string;
  updatedAt: string;
}

export interface Order {
  id: string; // client_order_id
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;
  type: "MARKET" | "LIMIT";
  limitPrice?: number;
  status: OrderStatus;
  submittedAt?: string;
  updatedAt: string;
  brokerOrderId?: string;
  strategyTag?: string;
  reason?: string;
}

export interface Fill {
  id: string;
  orderId: string;
  symbol: string;
  qty: number;
  price: number;
  ts: string;
  venue?: string;
  fees?: number;
}

export interface RiskMetrics {
  grossExposure: number;
  netExposure: number;
  maxDrawdownToday: number;
  feedStaleSeconds: number;
  rejectCount5m: number;
  slippageBpsAvg: number;
}

/**
 * WebSocket event envelope expected from FastAPI:
 * { type: "ORDERS", payload: Order[] }
 */
export type WSEvent =
  | { type: "HEARTBEAT"; payload?: any }
  | {
      type: "CONNECTION";
      payload: Partial<{
        apiUp: boolean;
        brokerAlive: boolean;
        marketDataAlive: boolean;
      }>;
    }
  | { type: "ENGINE_STATE"; payload: Partial<{ mode: EngineMode; runState: RunState; canTrade: boolean; warmupProgress: number; killSwitchArmed: boolean }> }
  | { type: "ACCOUNT_SNAPSHOT"; payload: AccountSnapshot }
  | { type: "POSITIONS"; payload: Position[] }
  | { type: "ORDERS"; payload: Order[] }
  | { type: "FILLS"; payload: Fill[] }
  | { type: "RISK_METRICS"; payload: RiskMetrics }
  | { type: string; payload: any }; // allow extension