import { useAppStore } from "../store";

export function RiskPage() {
  const m = useAppStore((s) => s.metrics);

  return (
    <div className="card">
      <div className="h2">Risk</div>
      {!m ? (
        <div className="muted">No risk metrics yet.</div>
      ) : (
        <div className="grid-3">
          <div className="card">
            <div className="muted">Gross Exposure</div>
            <div className="h1">{m.grossExposure}</div>
          </div>
          <div className="card">
            <div className="muted">Net Exposure</div>
            <div className="h1">{m.netExposure}</div>
          </div>
          <div className="card">
            <div className="muted">Max DD Today</div>
            <div className="h1">{m.maxDrawdownToday}</div>
          </div>
        </div>
      )}
    </div>
  );
}