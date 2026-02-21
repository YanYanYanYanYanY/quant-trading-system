import { useState } from "react";
import { useAppStore } from "../store";
import { placeOrder } from "../app/api";

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

export function OrdersPage() {
  const ids = useAppStore((s) => s.ids);
  const byId = useAppStore((s) => s.byId);
  const upsertOrders = useAppStore((s) => s.upsertOrders);
  const [symbol, setSymbol] = useState("AAPL");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [qty, setQty] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = await placeOrder({
        symbol: symbol.trim().toUpperCase(),
        side,
        qty,
      });
      upsertOrders([res.order]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Order failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Orders</h1>
      </div>

      <div className="card">
        <div className="h2">Place Order</div>
        <form
          onSubmit={handleSubmit}
          className="row"
          style={{ gap: 12, alignItems: "flex-end" }}
        >
          <label>
            Symbol
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="AAPL"
              style={{ width: 100 }}
            />
          </label>
          <label>
            Side
            <select
              value={side}
              onChange={(e) => setSide(e.target.value as "buy" | "sell")}
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </label>
          <label>
            Qty
            <input
              type="number"
              min={1}
              value={qty}
              onChange={(e) => setQty(Number(e.target.value) || 1)}
              style={{ width: 80 }}
            />
          </label>
          <button
            type="submit"
            className="button button-primary"
            disabled={busy}
          >
            {busy ? "Submitting\u2026" : "Place Order"}
          </button>
        </form>
        {error && (
          <div style={{ marginTop: 8, color: "var(--red)", fontSize: 12 }}>
            {error}
          </div>
        )}
      </div>

      <div className="card">
        <div className="h2">Order History</div>
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
              <tr>
                <td colSpan={7} className="muted">
                  No orders yet.
                </td>
              </tr>
            ) : (
              ids.slice(0, 200).map((id) => {
                const o = byId[id];
                if (!o) return null;
                return (
                  <tr key={id}>
                    <td className="muted">
                      {o.updatedAt?.slice(0, 19) ?? ""}
                    </td>
                    <td style={{ fontSize: 11 }}>{o.id}</td>
                    <td><b>{o.symbol}</b></td>
                    <td><SideBadge side={o.side} /></td>
                    <td>{o.qty}</td>
                    <td><StatusBadge status={o.status} /></td>
                    <td className="muted">{o.reason ?? ""}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
