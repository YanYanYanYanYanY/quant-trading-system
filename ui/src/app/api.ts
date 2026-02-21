/**
 * REST client for Quant Trading API.
 * Base URL: VITE_API_BASE_URL or http://localhost:8000
 */

import type { Order, Position } from "./types";

const getBase = () =>
  (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000";

export function getApiBase(): string {
  return getBase().replace(/\/$/, "");
}

export function getWsUrl(): string {
  const env = import.meta.env.VITE_WS_URL as string | undefined;
  if (env) return env;
  const base = getBase();
  const wsProtocol = base.startsWith("https") ? "wss" : "ws";
  const host = base.replace(/^https?:\/\//, "");
  return `${wsProtocol}://${host}/ws`;
}

// --- Health ---

export async function checkHealth(): Promise<{ ok: boolean }> {
  const r = await fetch(`${getApiBase()}/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

// --- Trade status ---

export type ApiStatus = {
  mode: string;
  engine_ok: boolean;
  ws_clients: number;
  last_event_ts: string | null;
};

export async function getTradeStatus(): Promise<ApiStatus> {
  const r = await fetch(`${getApiBase()}/trade/status`);
  if (!r.ok) throw new Error(`Status failed: ${r.status}`);
  return r.json();
}

// --- Positions (API returns list of dicts: symbol, qty, avg_price, etc.) ---

function normalizePosition(p: Record<string, unknown>): Position {
  const qty = Number(p.qty ?? 0);
  const avgPrice = Number(p.avg_price ?? p.avgPrice ?? 0);
  const marketPrice = Number(p.market_price ?? p.marketPrice ?? avgPrice);
  return {
    symbol: String(p.symbol ?? ""),
    qty,
    avgPrice,
    marketPrice,
    unrealizedPnl: Number(p.unrealized_pnl ?? p.unrealizedPnl ?? 0),
    strategyTag: p.strategy_tag != null ? String(p.strategy_tag) : p.strategyTag != null ? String(p.strategyTag) : undefined,
    updatedAt: (p.updated_at ?? p.updatedAt ?? new Date().toISOString()) as string,
  };
}

export async function getPositions(): Promise<Position[]> {
  const r = await fetch(`${getApiBase()}/trade/positions`);
  if (!r.ok) throw new Error(`Positions failed: ${r.status}`);
  const data = await r.json();
  const list = Array.isArray(data.positions) ? data.positions : [];
  return list.map(normalizePosition);
}

// --- Orders (API returns list of dicts: id, symbol, side, qty, type, limit_price, status, ts) ---

export function normalizeOrder(o: Record<string, unknown>): Order {
  const side = String(o.side ?? "BUY").toUpperCase() as "BUY" | "SELL";
  const type = String(o.type ?? "MARKET").toUpperCase() as "MARKET" | "LIMIT";
  const ts = (o.ts ?? o.updated_at ?? o.updatedAt ?? new Date().toISOString()) as string;
  return {
    id: String(o.id ?? ""),
    symbol: String(o.symbol ?? ""),
    side: side === "SELL" ? "SELL" : "BUY",
    qty: Number(o.qty ?? 0),
    type: type === "LIMIT" ? "LIMIT" : "MARKET",
    limitPrice: o.limit_price != null ? Number(o.limit_price) : o.limitPrice != null ? Number(o.limitPrice) : undefined,
    status: String(o.status ?? "NEW").toUpperCase() as Order["status"],
    submittedAt: (o.submitted_at ?? o.submittedAt) as string | undefined,
    updatedAt: ts,
    brokerOrderId: o.broker_order_id != null ? String(o.broker_order_id) : o.brokerOrderId != null ? String(o.brokerOrderId) : undefined,
    strategyTag: o.strategy_tag != null ? String(o.strategy_tag) : o.strategyTag != null ? String(o.strategyTag) : undefined,
    reason: o.reason != null ? String(o.reason) : undefined,
  };
}

export async function getOrders(): Promise<Order[]> {
  const r = await fetch(`${getApiBase()}/trade/orders`);
  if (!r.ok) throw new Error(`Orders failed: ${r.status}`);
  const data = await r.json();
  const list = Array.isArray(data.orders) ? data.orders : [];
  return list.map(normalizeOrder);
}

// --- Strategies ---

export async function getStrategies(): Promise<string[]> {
  const r = await fetch(`${getApiBase()}/strategies`);
  if (!r.ok) throw new Error(`Strategies failed: ${r.status}`);
  const data = await r.json();
  return Array.isArray(data.strategies) ? data.strategies : [];
}

// --- Engine controls ---

export async function startTrading(): Promise<ApiStatus> {
  const r = await fetch(`${getApiBase()}/trade/start`, { method: "POST" });
  if (!r.ok) throw new Error(`Start failed: ${r.status}`);
  return r.json();
}

export async function stopTrading(): Promise<ApiStatus> {
  const r = await fetch(`${getApiBase()}/trade/stop`, { method: "POST" });
  if (!r.ok) throw new Error(`Stop failed: ${r.status}`);
  return r.json();
}

// --- Place order ---

export type PlaceOrderBody = {
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  order_type?: "market" | "limit";
  limit_price?: number;
};

export async function placeOrder(body: PlaceOrderBody): Promise<{ order: Order }> {
  const r = await fetch(`${getApiBase()}/trade/orders`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: body.symbol,
      side: body.side,
      qty: body.qty,
      order_type: body.order_type ?? "market",
      limit_price: body.limit_price ?? null,
    }),
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(text || `Place order failed: ${r.status}`);
  }
  const data = await r.json();
  return { order: normalizeOrder(data.order ?? data) };
}

// --- Account ---

export type ApiAccount = {
  equity: number | null;
  cash: number | null;
  buying_power: number | null;
  portfolio_value: number | null;
  daily_pnl: number | null;
};

export async function getAccount(): Promise<ApiAccount> {
  const r = await fetch(`${getApiBase()}/trade/account`);
  if (!r.ok) throw new Error(`Account failed: ${r.status}`);
  return r.json();
}

// --- Market clock ---

export type ApiClock = {
  is_open: boolean;
  next_open: string | null;
  next_close: string | null;
  timestamp?: string | null;
};

export async function getClock(): Promise<ApiClock> {
  const r = await fetch(`${getApiBase()}/trade/clock`);
  if (!r.ok) throw new Error(`Clock failed: ${r.status}`);
  return r.json();
}

// --- Flatten all ---

export async function flattenAll(): Promise<{ cancelled: number; closed: number }> {
  const r = await fetch(`${getApiBase()}/trade/flatten`, { method: "POST" });
  if (!r.ok) throw new Error(`Flatten failed: ${r.status}`);
  return r.json();
}

// --- Map API status to store engine state ---

export function statusToEngine(p: ApiStatus): {
  mode: "BACKTEST" | "PAPER" | "LIVE";
  runState: "STOPPED" | "RUNNING" | "PAUSED" | "KILLED";
  canTrade: boolean;
} {
  const mode = (p.mode === "live" ? "LIVE" : p.mode === "paper" ? "PAPER" : "BACKTEST") as "BACKTEST" | "PAPER" | "LIVE";
  const runState = (p.mode === "stopped" ? "STOPPED" : "RUNNING") as "STOPPED" | "RUNNING" | "PAUSED" | "KILLED";
  return {
    mode,
    runState,
    canTrade: runState === "RUNNING" && p.engine_ok,
  };
}
