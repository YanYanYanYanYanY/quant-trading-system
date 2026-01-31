import type { ReactNode } from "react";
import { SidebarNav } from "./SidebarNav";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
          <div>
            <div className="h1">Quant GUI</div>
            <div className="muted" style={{ fontSize: 12 }}>private dashboard</div>
          </div>
        </div>
        <SidebarNav />
      </aside>

      <section className="main">
        <header className="topbar">
          <TopBar />
        </header>
        <main className="container">{children}</main>
      </section>
    </div>
  );
}