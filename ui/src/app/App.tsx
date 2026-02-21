import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import { DashboardPage } from "../pages/DashboardPage";
import { OrdersPage } from "../pages/OrdersPage";
import { PositionsPage } from "../pages/PositionsPage";
import { StrategiesPage } from "../pages/StrategiesPage";
import { RiskPage } from "../pages/RiskPage";
import { LogsPage } from "../pages/LogsPage";
import { useTradingWS } from "./ws";
import { useApiSync } from "./useApiSync";
import { getWsUrl } from "./api";

export default function App() {
  useApiSync();
  useTradingWS(getWsUrl());

  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/strategies" element={<StrategiesPage />} />
          <Route path="/orders" element={<OrdersPage />} />
          <Route path="/positions" element={<PositionsPage />} />
          <Route path="/risk" element={<RiskPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}