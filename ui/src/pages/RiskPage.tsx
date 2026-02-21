import { useAppStore } from "../store";

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent: string;
}) {
  return (
    <div className="card" style={{ borderLeft: `3px solid var(--${accent})` }}>
      <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
        {label}
      </div>
      <div className="h1">{value}</div>
    </div>
  );
}

export function RiskPage() {
  const m = useAppStore((s) => s.metrics);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Risk</h1>
      </div>

      <div className="card">
        <div className="h2">Risk Metrics</div>
        {!m ? (
          <div className="muted">No risk metrics yet.</div>
        ) : (
          <div className="grid-3">
            <MetricCard
              label="Gross Exposure"
              value={m.grossExposure}
              accent="blue"
            />
            <MetricCard
              label="Net Exposure"
              value={m.netExposure}
              accent="accent"
            />
            <MetricCard
              label="Max DD Today"
              value={m.maxDrawdownToday}
              accent="red"
            />
          </div>
        )}
      </div>
    </div>
  );
}
