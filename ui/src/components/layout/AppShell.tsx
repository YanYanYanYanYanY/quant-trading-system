import type { ReactNode } from "react";
import { SidebarNav } from "./SidebarNav";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <TopBar />
      </header>

      <aside className="sidebar">
        <SidebarNav />
      </aside>

      <main className="main">{children}</main>
    </div>
  );
}
