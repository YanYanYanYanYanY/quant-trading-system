import { useEffect } from "react";
import { useAppStore } from "../store";

export function MockEventFeeder() {
  const applyEvent = useAppStore((s) => s.applyEvent);

  useEffect(() => {
    // heartbeat
    const hb = setInterval(() => {
      applyEvent({ type: "HEARTBEAT" });
    }, 1000);

    // fake account snapshot
    applyEvent({
      type: "ACCOUNT_SNAPSHOT",
      payload: {
        equity: 100_000,
        cash: 60_000,
        buyingPower: 120_000,
        dailyPnl: 350,
        updatedAt: new Date().toISOString(),
      },
    });

    // fake order stream
    let i = 0;
    const ord = setInterval(() => {
      i++;
      applyEvent({
        type: "ORDERS",
        payload: [
          {
            id: `MOCK-${i}`,
            symbol: i % 2 ? "AAPL" : "MSFT",
            side: i % 2 ? "BUY" : "SELL",
            qty: 100,
            type: "MARKET",
            status: "FILLED",
            updatedAt: new Date().toISOString(),
            strategyTag: "mean_revert",
          },
        ],
      });
    }, 2000);

    return () => {
      clearInterval(hb);
      clearInterval(ord);
    };
  }, [applyEvent]);

  return null;
}