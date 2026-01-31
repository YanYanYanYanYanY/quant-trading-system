import { useEffect, useRef } from "react";
import { useAppStore } from "../store";
import type { WSEvent } from "./types";

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
        const evt = JSON.parse(msg.data) as WSEvent;
        applyEvent(evt);
      } catch {
        // ignore malformed
      }
    };

    return () => ws.close();
  }, [url, applyEvent, setConnection]);
}