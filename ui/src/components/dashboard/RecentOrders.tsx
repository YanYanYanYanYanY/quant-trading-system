import { useAppStore } from "../../store";
import { selectTopOrders } from "../../store/selectors/dashboard";
import { useShallow } from "zustand/react/shallow";

export function RecentOrders() {
  const orders = useAppStore(
    useShallow((s) => selectTopOrders(s, 12))
  );

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="h2">Recent Orders</div>
        <div className="muted" style={{ fontSize: 12 }}>showing latest 12</div>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Status</th>
            <th>Strategy</th>
          </tr>
        </thead>
        <tbody>
          {orders.length === 0 ? (
            <tr><td colSpan={6} className="muted">No orders yet.</td></tr>
          ) : (
            orders.map((o) => (
              <tr key={o.id}>
                <td>{o.updatedAt?.slice(11, 19) ?? ""}</td>
                <td><b>{o.symbol}</b></td>
                <td>{o.side}</td>
                <td>{o.qty}</td>
                <td>{o.status}</td>
                <td>{o.strategyTag ?? "—"}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}