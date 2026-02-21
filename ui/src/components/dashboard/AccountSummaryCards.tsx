import { useAppStore } from "../../store";

function fmt(n?: number) {
  if (n === undefined || n === null) return "\u2014";
  return n.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

export function AccountSummaryCards() {
  const snap = useAppStore((s) => s.snapshot);

  const pnl = snap?.dailyPnl ?? 0;
  const pnlColor = pnl > 0 ? "text-green" : pnl < 0 ? "text-red" : "";

  return (
    <div className="grid-3">
      <div className="card" style={{ borderLeft: "3px solid var(--accent)" }}>
        <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>Equity</div>
        <div className="h1">{fmt(snap?.equity)}</div>
        <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
          {snap?.updatedAt ?? ""}
        </div>
      </div>

      <div className="card" style={{ borderLeft: "3px solid var(--blue)" }}>
        <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>Cash</div>
        <div className="h1">{fmt(snap?.cash)}</div>
        <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
          Buying Power: {fmt(snap?.buyingPower)}
        </div>
      </div>

      <div className="card" style={{ borderLeft: `3px solid var(${pnl >= 0 ? "--green" : "--red"})` }}>
        <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>Daily PnL</div>
        <div className={`h1 ${pnlColor}`}>{fmt(snap?.dailyPnl)}</div>
        <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>Real-time via WS</div>
      </div>
    </div>
  );
}
