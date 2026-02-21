import { useState } from "react";
import { useAppStore } from "../../store";
import { startTrading, stopTrading, statusToEngine } from "../../app/api";

export function TradeControls() {
  const runState = useAppStore((s) => s.runState);
  const setEngine = useAppStore((s) => s.setEngine);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await startTrading();
      setEngine(statusToEngine(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Start failed");
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await stopTrading();
      setEngine(statusToEngine(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stop failed");
    } finally {
      setBusy(false);
    }
  };

  const running = runState === "RUNNING";

  const stateClass =
    running ? "badge-green" : runState === "PAUSED" ? "badge-amber" : "badge-blue";

  return (
    <div className="card">
      <div className="h2">Engine</div>
      <div className="row" style={{ gap: 10 }}>
        <span className={`badge ${stateClass}`}>
          <span
            className={`badge-dot ${
              running ? "badge-dot-green" : runState === "PAUSED" ? "badge-dot-amber" : "badge-dot-blue"
            }`}
          />
          {runState}
        </span>
        {!running ? (
          <button
            type="button"
            className="button button-green"
            onClick={handleStart}
            disabled={busy}
          >
            {busy ? "Starting\u2026" : "Start Trading"}
          </button>
        ) : (
          <button
            type="button"
            className="button button-danger"
            onClick={handleStop}
            disabled={busy}
          >
            {busy ? "Stopping\u2026" : "Stop Trading"}
          </button>
        )}
      </div>
      {error && (
        <div style={{ marginTop: 8, color: "var(--red)", fontSize: 12 }}>
          {error}
        </div>
      )}
    </div>
  );
}
