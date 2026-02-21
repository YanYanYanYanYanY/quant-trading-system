import { useAppStore } from "../../store";
import { selectHealth } from "../../store/selectors/dashboard";
import { useShallow } from "zustand/react/shallow";

function Dot({ ok }: { ok: boolean | string }) {
  const isOk = ok === true || ok === "true";
  return (
    <span
      className={`badge-dot ${isOk ? "badge-dot-green" : "badge-dot-red"}`}
    />
  );
}

export function StatusStrip() {
  const health = useAppStore(useShallow(selectHealth));

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="h2" style={{ margin: 0 }}>System Status</div>
        <span className="badge" style={{ fontSize: 11 }}>
          Heartbeat:{" "}
          <b>
            {health.hbAgeSeconds === null
              ? "n/a"
              : `${health.hbAgeSeconds.toFixed(1)}s`}
          </b>
        </span>
      </div>

      <div className="row" style={{ marginTop: 10 }}>
        <span className={`badge ${health.apiUp ? "badge-green" : "badge-red"}`}>
          <Dot ok={health.apiUp} /> API
        </span>
        <span className={`badge ${health.ws ? "badge-green" : "badge-red"}`}>
          <Dot ok={health.ws} /> WebSocket
        </span>
        <span className={`badge ${health.broker ? "badge-green" : "badge-amber"}`}>
          <Dot ok={health.broker} /> Broker
        </span>
        <span className={`badge ${health.data ? "badge-green" : "badge-amber"}`}>
          <Dot ok={health.data} /> Data
        </span>
        {health.heartbeatStale && (
          <span className="badge badge-red">
            <Dot ok={false} /> Stale
          </span>
        )}
      </div>
    </div>
  );
}
