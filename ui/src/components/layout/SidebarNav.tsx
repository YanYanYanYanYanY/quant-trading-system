import { NavLink } from "react-router-dom";

const items = [
  { to: "/", label: "Dashboard", icon: "\u{1F4CA}" },
  { to: "/strategies", label: "Strategies", icon: "\u{1F9E0}" },
  { to: "/orders", label: "Orders", icon: "\u{1F4DD}" },
  { to: "/positions", label: "Positions", icon: "\u{1F4BC}" },
  { to: "/risk", label: "Risk", icon: "\u{1F6E1}\uFE0F" },
  { to: "/logs", label: "Logs", icon: "\u{1F4C4}" },
];

export function SidebarNav() {
  return (
    <nav className="sidebar-nav">
      {items.map((it) => (
        <NavLink
          key={it.to}
          to={it.to}
          end={it.to === "/"}
          className={({ isActive }) =>
            `navlink${isActive ? " active" : ""}`
          }
        >
          <span className="navlink-icon">{it.icon}</span>
          {it.label}
        </NavLink>
      ))}
    </nav>
  );
}
