import { useAppStore } from "../../store";
import { selectTopOrders } from "../../store/selectors/dashboard";
import { useShallow } from "zustand/react/shallow";

function SideBadge({ side }: { side?: string }) {
  const s = side?.toLowerCase();
  const cls = s === "buy" ? "badge-green" : s === "sell" ? "badge-red" : "";
  return <span className={`badge ${cls}`} style={{ fontSize: 11 }}>{side ?? ""}</span>;
}

function StatusBadge({ status }: { status?: string }) {
  const s = status?.toLowerCase();
  let cls = "";
  if (s === "filled") cls = "badge-green";
  else if (s === "rejected" || s === "cancelled") cls = "badge-red";
  else if (s === "pending" || s === "new") cls = "badge-amber";
  else cls = "badge-blue";
  return <span className={`badge ${cls}`} style={{ fontSize: 11 }}>{status ?? ""}</span>;
}

export function RecentOrders() {
  const orders = useAppStore(useShallow((s) => selectTopOrders(s, 12)));

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
        <div className="h2" style={{ margin: 0 }}>Recent Orders</div>
        <span className="muted" style={{ fontSize: 11 }}>Latest 12</span>
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
            <tr>
              <td colSpan={6} className="muted">
                No orders yet.
              </td>
            </tr>
          ) : (
            orders.map((o) => (
              <tr key={o.id}>
                <td className="muted">{o.updatedAt?.slice(11, 19) ?? ""}</td>
                <td><b>{o.symbol}</b></td>
                <td><SideBadge side={o.side} /></td>
                <td>{o.qty}</td>
                <td><StatusBadge status={o.status} /></td>
                <td className="muted">{o.strategyTag ?? "\u2014"}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
