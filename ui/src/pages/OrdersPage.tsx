import { useAppStore } from "../store";

export function OrdersPage() {
  const ids = useAppStore((s) => s.ids);
  const byId = useAppStore((s) => s.byId);

  return (
    <div className="card">
      <div className="h2">Orders</div>
      <table className="table">
        <thead>
          <tr>
            <th>Updated</th>
            <th>Id</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Qty</th>
            <th>Status</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {ids.length === 0 ? (
            <tr><td colSpan={7} className="muted">No orders yet.</td></tr>
          ) : (
            ids.slice(0, 200).map((id) => {
              const o = byId[id];
              if (!o) return null;
              return (
                <tr key={id}>
                  <td>{o.updatedAt?.slice(0, 19) ?? ""}</td>
                  <td>{o.id}</td>
                  <td><b>{o.symbol}</b></td>
                  <td>{o.side}</td>
                  <td>{o.qty}</td>
                  <td>{o.status}</td>
                  <td className="muted">{o.reason ?? ""}</td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}