import { useShallow } from "zustand/react/shallow";
import { useAppStore } from "../store";
import type { Position } from "../app/types";

function PnlCell({ value }: { value?: number }) {
  if (value == null) return <span className="muted">{"\u2014"}</span>;
  const cls = value > 0 ? "text-green" : value < 0 ? "text-red" : "";
  const prefix = value > 0 ? "+" : "";
  return (
    <b className={cls}>
      {prefix}
      {value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
    </b>
  );
}

function fmt(n?: number | null) {
  if (n == null) return "\u2014";
  return n.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

function selectPositions(s: { bySymbol: Record<string, Position> }): Position[] {
  return Object.values(s.bySymbol ?? {});
}

export function PositionsPage() {
  const positions = useAppStore(useShallow(selectPositions));
  const equity = useAppStore((s) => s.snapshot?.equity);
  const cash = useAppStore((s) => s.snapshot?.cash);
  const buyingPower = useAppStore((s) => s.snapshot?.buyingPower);
  const runState = useAppStore((s) => s.runState);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Positions</h1>
        <span className="badge badge-blue">{positions.length} open</span>
      </div>

      {(equity != null || cash != null || buyingPower != null) && (
        <div className="grid-3">
          <div
            className="card"
            style={{ borderLeft: "3px solid var(--accent)" }}
          >
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              Equity
            </div>
            <div className="h1">{fmt(equity)}</div>
          </div>
          <div
            className="card"
            style={{ borderLeft: "3px solid var(--blue)" }}
          >
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              Cash
            </div>
            <div className="h1">{fmt(cash)}</div>
          </div>
          <div
            className="card"
            style={{ borderLeft: "3px solid var(--green)" }}
          >
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              Buying Power
            </div>
            <div className="h1">{fmt(buyingPower)}</div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="h2">Open Positions</div>
        <table className="table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Qty</th>
              <th>Avg Price</th>
              <th>Mkt Price</th>
              <th>Unrealized PnL</th>
              <th>Strategy</th>
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 ? (
              <tr>
                <td colSpan={6} className="muted">
                  {runState === "RUNNING"
                    ? "No open positions. Place an order to get started."
                    : "No open positions. Start trading first, then place orders."}
                </td>
              </tr>
            ) : (
              positions.map((p) => (
                <tr key={p.symbol}>
                  <td>
                    <b>{p.symbol}</b>
                  </td>
                  <td>{p.qty}</td>
                  <td>{p.avgPrice}</td>
                  <td>{p.marketPrice}</td>
                  <td>
                    <PnlCell value={p.unrealizedPnl} />
                  </td>
                  <td className="muted">{p.strategyTag ?? "\u2014"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
