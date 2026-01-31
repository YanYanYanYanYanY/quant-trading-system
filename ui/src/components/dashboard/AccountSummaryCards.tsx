import { useAppStore } from "../../store";

function fmt(n?: number) {
  if (n === undefined || n === null) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function AccountSummaryCards() {
  const snap = useAppStore((s) => s.snapshot);

  return (
    <div className="grid-3">
      <div className="card">
        <div className="muted">Equity</div>
        <div className="h1">{fmt(snap?.equity)}</div>
        <div className="muted" style={{ fontSize: 12 }}>{snap?.updatedAt ?? ""}</div>
      </div>

      <div className="card">
        <div className="muted">Cash</div>
        <div className="h1">{fmt(snap?.cash)}</div>
        <div className="muted" style={{ fontSize: 12 }}>Buying Power: {fmt(snap?.buyingPower)}</div>
      </div>

      <div className="card">
        <div className="muted">Daily PnL</div>
        <div className="h1">{fmt(snap?.dailyPnl)}</div>
        <div className="muted" style={{ fontSize: 12 }}>Real-time via WS</div>
      </div>
    </div>
  );
}