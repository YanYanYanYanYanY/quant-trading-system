import { StatusStrip } from "../components/dashboard/StatusStrip";
import { AccountSummaryCards } from "../components/dashboard/AccountSummaryCards";
import { RecentOrders } from "../components/dashboard/RecentOrders";
import { TradeControls } from "../components/dashboard/TradeControls";

export function DashboardPage() {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
      </div>
      <StatusStrip />
      <TradeControls />
      <AccountSummaryCards />
      <div className="grid-2">
        <RecentOrders />
        <div className="card">
          <div className="h2">Coming Soon</div>
          <div className="muted" style={{ lineHeight: 1.6 }}>
            Exposure panel, PnL chart, recent signals, risk alerts.
          </div>
        </div>
      </div>
    </div>
  );
}
