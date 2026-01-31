import { useAppStore } from "../store";

export function PositionsPage() {
  const positions = useAppStore((s) => Object.values(s.bySymbol));

  return (
    <div className="card">
      <div className="h2">Positions</div>
      <table className="table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Qty</th>
            <th>Avg</th>
            <th>Mkt</th>
            <th>UPnL</th>
            <th>Strategy</th>
          </tr>
        </thead>
        <tbody>
          {positions.length === 0 ? (
            <tr><td colSpan={6} className="muted">No positions.</td></tr>
          ) : (
            positions.map((p) => (
              <tr key={p.symbol}>
                <td><b>{p.symbol}</b></td>
                <td>{p.qty}</td>
                <td>{p.avgPrice}</td>
                <td>{p.marketPrice}</td>
                <td>{p.unrealizedPnl}</td>
                <td className="muted">{p.strategyTag ?? "—"}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}