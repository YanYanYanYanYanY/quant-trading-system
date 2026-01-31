import { NavLink } from "react-router-dom";

const items = [
    { to: "/", label: "Dashboard" },
    { to: "/strategies", label: "Strategies" },
    { to: "/orders", label: "Orders" },
    { to: "/positions", label: "Positions" },
    { to: "/risk", label: "Risk" },
    { to: "/logs", label: "Logs" },
];

export function SidebarNav() {
    return (
        <nav className="row" style={{ flexDirection: "column", gap: 6 }}>
            {items.map((it) => (
                <NavLink
                    key={it.to}
                    to={it.to}
                    className={({ isActive }) => `navlink ${isActive ? "active" : ""}`}
                >
                    {it.label}
                </NavLink>
            ))}
        </nav>
    );
}