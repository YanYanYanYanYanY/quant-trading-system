export function LogsPage() {
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="page-header">
        <h1 className="page-title">Logs</h1>
      </div>

      <div className="card">
        <div className="h2">Event Log</div>
        <div
          style={{
            padding: 24,
            textAlign: "center",
            background: "var(--content-bg)",
            borderRadius: 8,
            border: "1px dashed var(--border-light)",
          }}
        >
          <div className="muted" style={{ fontSize: 13 }}>
            WebSocket log stream, filters, and export will appear here.
          </div>
        </div>
      </div>
    </div>
  );
}
