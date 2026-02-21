import { useEffect, useRef } from "react";
import { useAppStore } from "../store";
import { normalizeOrder } from "./api";
import type { WSEvent } from "./types";

/** Normalize API WS events (e.g. order_submitted with .data) into store events (ORDERS with .payload). */
function toStoreEvent(raw: Record<string, unknown>): WSEvent {
  if (raw.type === "order_submitted" && raw.data != null) {
    return { type: "ORDERS", payload: [normalizeOrder(raw.data as Record<string, unknown>)] };
  }
  return raw as WSEvent;
}

export function useTradingWS(url: string) {
  const applyEvent = useAppStore((s) => s.applyEvent);
  const setConnection = useAppStore((s) => s.setConnection);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnection({ wsConnected: true });
    ws.onclose = () => setConnection({ wsConnected: false });
    ws.onerror = () => setConnection({ wsConnected: false });

    ws.onmessage = (msg) => {
      try {
        const raw = JSON.parse(msg.data) as Record<string, unknown>;
        const evt = toStoreEvent(raw);
        applyEvent(evt);
      } catch {
        // ignore malformed
      }
    };

    return () => ws.close();
  }, [url, applyEvent, setConnection]);
}