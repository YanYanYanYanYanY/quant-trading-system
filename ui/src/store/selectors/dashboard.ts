import type { AppStore } from "../index";

export function selectHealth(s: AppStore) {
  const now = Date.now();
  const hbAge = s.lastHeartbeatTs ? (now - s.lastHeartbeatTs) / 1000 : Infinity;

  return {
    apiUp: s.apiUp,
    ws: s.wsConnected,
    broker: s.brokerAlive,
    data: s.marketDataAlive,
    heartbeatStale: hbAge > 5,
    hbAgeSeconds: isFinite(hbAge) ? hbAge : null,
  };
}

export function selectTopOrders(s: AppStore, n = 12) {
  return s.ids.slice(0, n).map((id) => s.byId[id]).filter(Boolean);
}