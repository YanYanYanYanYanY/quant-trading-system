import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "../components/layout/AppShell";
import { DashboardPage } from "../pages/DashboardPage";
import { OrdersPage } from "../pages/OrdersPage";
import { PositionsPage } from "../pages/PositionsPage";
import { StrategiesPage } from "../pages/StrategiesPage";
import { RiskPage } from "../pages/RiskPage";
import { LogsPage } from "../pages/LogsPage";
import { useTradingWS } from "./ws";
import { MockEventFeeder } from "./mockEvents";


export default function App() {
  // change to your FastAPI websocket endpoint
  useTradingWS("ws://localhost:8000/ws");

  return (

    <BrowserRouter>
      <AppShell>
        <>
          <MockEventFeeder />
        </>
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