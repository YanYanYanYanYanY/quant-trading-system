import { useEffect, useRef } from "react";
import { useAppStore } from "../store";
import {
  checkHealth,
  getTradeStatus,
  getPositions,
  getOrders,
  getAccount,
  statusToEngine,
} from "./api";
import type { AccountSnapshot } from "./types";

const HEALTH_INTERVAL_MS = 5000;
const DATA_REFRESH_MS = 10000;

/**
 * Polls /health and sets apiUp in the store.
 * When API is up, fetches status, positions, orders, and account
 * and syncs everything to the Zustand store on an interval.
 */
export function useApiSync() {
  const setConnection = useAppStore((s) => s.setConnection);
  const setEngine = useAppStore((s) => s.setEngine);
  const setAccount = useAppStore((s) => s.setAccount);
  const upsertPositions = useAppStore((s) => s.upsertPositions);
  const upsertOrders = useAppStore((s) => s.upsertOrders);
  const apiUp = useAppStore((s) => s.apiUp);
  const dataIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Health poll
  useEffect(() => {
    let mounted = true;

    const poll = async () => {
      try {
        const res = await checkHealth();
        if (mounted) setConnection({ apiUp: !!res?.ok });
      } catch {
        if (mounted) setConnection({ apiUp: false });
      }
    };

    poll();
    const t = setInterval(poll, HEALTH_INTERVAL_MS);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, [setConnection]);

  // When API is up: fetch status, positions, orders, account; then refresh
  useEffect(() => {
    if (!apiUp) {
      if (dataIntervalRef.current) {
        clearInterval(dataIntervalRef.current);
        dataIntervalRef.current = null;
      }
      return;
    }

    let mounted = true;

    const fetchAll = async () => {
      try {
        const [statusRes, positionsRes, ordersRes, accountRes] =
          await Promise.all([
            getTradeStatus(),
            getPositions(),
            getOrders(),
            getAccount(),
          ]);
        if (!mounted) return;

        setEngine(statusToEngine(statusRes));
        upsertPositions(positionsRes);
        upsertOrders(ordersRes);

        const snap: AccountSnapshot = {
          equity: accountRes.equity ?? 0,
          cash: accountRes.cash ?? 0,
          buyingPower: accountRes.buying_power ?? 0,
          dailyPnl: accountRes.daily_pnl ?? 0,
          updatedAt: new Date().toISOString(),
        };
        setAccount(snap);
      } catch {
        // keep previous state on transient errors
      }
    };

    fetchAll();
    dataIntervalRef.current = setInterval(fetchAll, DATA_REFRESH_MS);

    return () => {
      mounted = false;
      if (dataIntervalRef.current) {
        clearInterval(dataIntervalRef.current);
        dataIntervalRef.current = null;
      }
    };
  }, [apiUp, setEngine, setAccount, upsertPositions, upsertOrders]);
}
