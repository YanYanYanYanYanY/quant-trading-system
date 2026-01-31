import { useAppStore } from "../../store";
import { selectHealth } from "../../store/selectors/dashboard";
import { useShallow } from "zustand/react/shallow";

export function StatusStrip() {
  const health = useAppStore(useShallow(selectHealth));

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="h2">System Status</div>
        <div className="muted" style={{ fontSize: 12 }}>
          heartbeat age:{" "}
          {health.hbAgeSeconds === null
            ? "n/a"
            : `${health.hbAgeSeconds.toFixed(1)}s`}
        </div>
      </div>

      <div className="row">
        <span className="badge">API: <b>{String(health.apiUp)}</b></span>
        <span className="badge">WS: <b>{String(health.ws)}</b></span>
        <span className="badge">Broker: <b>{String(health.broker)}</b></span>
        <span className="badge">Data: <b>{String(health.data)}</b></span>
        <span className="badge">Stale: <b>{String(health.heartbeatStale)}</b></span>
      </div>
    </div>
  );
}