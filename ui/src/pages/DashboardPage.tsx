import { StatusStrip } from "../components/dashboard/StatusStrip";
import { AccountSummaryCards } from "../components/dashboard/AccountSummaryCards";
import { RecentOrders } from "../components/dashboard/RecentOrders";

export function DashboardPage() {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <StatusStrip />
      <AccountSummaryCards />
      <div className="grid-2">
        <RecentOrders />
        <div className="card">
          <div className="h2">Placeholder</div>
          <div className="muted">
            Add: exposure panel, PnL chart, recent signals, risk alerts.
          </div>
        </div>
      </div>
    </div>
  );
}