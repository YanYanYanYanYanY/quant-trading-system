import { useEffect, useState } from "react";
import { getStrategies } from "../app/api";

export function StrategiesPage() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    getStrategies()
      .then((list) => {
        if (mounted) setStrategies(list);
      })
      .catch((e) => {
        if (mounted)
          setError(e instanceof Error ? e.message : "Failed to load");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Strategies</h1>
      </div>

      <div className="card">
        <div className="h2">Active Strategies</div>
        {loading && <div className="muted">Loading\u2026</div>}
        {error && (
          <div style={{ color: "var(--red)", fontSize: 13 }}>{error}</div>
        )}
        {!loading && !error && (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {strategies.length === 0 ? (
              <div className="muted">No strategies available.</div>
            ) : (
              strategies.map((name) => (
                <div
                  key={name}
                  className="row"
                  style={{
                    padding: "8px 12px",
                    background: "var(--content-bg)",
                    borderRadius: 8,
                    border: "1px solid var(--border-light)",
                  }}
                >
                  <span className="badge badge-purple" style={{ fontSize: 11 }}>
                    Active
                  </span>
                  <b>{name}</b>
                </div>
              ))
            )}
          </div>
        )}
        <div
          className="muted"
          style={{ marginTop: 14, fontSize: 12, fontStyle: "italic" }}
        >
          Enable toggles and parameter editor can be added here.
        </div>
      </div>
    </div>
  );
}
