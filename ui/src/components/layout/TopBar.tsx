import { useState } from "react";
import { useAppStore } from "../../store";
import {
  startTrading,
  stopTrading,
  flattenAll,
  statusToEngine,
} from "../../app/api";

function ModeBadge() {
  const mode = useAppStore((s) => s.mode);
  const runState = useAppStore((s) => s.runState);

  const stateClass =
    runState === "RUNNING"
      ? "badge-green"
      : runState === "PAUSED"
        ? "badge-amber"
        : "badge-blue";

  return (
    <span className={`badge ${stateClass}`}>
      <span
        className={`badge-dot ${
          runState === "RUNNING"
            ? "badge-dot-green"
            : runState === "PAUSED"
              ? "badge-dot-amber"
              : "badge-dot-blue"
        }`}
      />
      {mode} &middot; {runState}
    </span>
  );
}

export function TopBar() {
  const runState = useAppStore((s) => s.runState);
  const canTrade = useAppStore((s) => s.canTrade);
  const killSwitchArmed = useAppStore((s) => s.killSwitchArmed);
  const setEngine = useAppStore((s) => s.setEngine);

  const [startBusy, setStartBusy] = useState(false);
  const [stopBusy, setStopBusy] = useState(false);
  const [flattenBusy, setFlattenBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const running = runState === "RUNNING";

  const handleStart = async () => {
    setError(null);
    setStartBusy(true);
    try {
      const res = await startTrading();
      setEngine(statusToEngine(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Start failed");
    } finally {
      setStartBusy(false);
    }
  };

  const handleStop = async () => {
    setError(null);
    setStopBusy(true);
    try {
      const res = await stopTrading();
      setEngine(statusToEngine(res));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Stop failed");
    } finally {
      setStopBusy(false);
    }
  };

  const handleFlatten = async () => {
    setError(null);
    setFlattenBusy(true);
    try {
      await flattenAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Flatten failed");
    } finally {
      setFlattenBusy(false);
    }
  };

  return (
    <>
      <div className="topbar-brand">
        <div>
          <div className="topbar-brand-title">Quant Trading</div>
          <div className="topbar-brand-sub">Private Dashboard</div>
        </div>
        <div className="topbar-divider" />
      </div>

      <div className="topbar-center">
        <ModeBadge />
        <span className={`badge ${canTrade ? "badge-green" : "badge-amber"}`}>
          <span
            className={`badge-dot ${canTrade ? "badge-dot-green" : "badge-dot-amber"}`}
          />
          Trade {canTrade ? "Enabled" : "Disabled"}
        </span>
        {killSwitchArmed && (
          <span className="badge badge-red">
            <span className="badge-dot badge-dot-red" />
            Kill Switch ARMED
          </span>
        )}
        {error && (
          <span
            style={{ color: "var(--red)", fontSize: 12, marginLeft: 4 }}
          >
            {error}
          </span>
        )}
      </div>

      <div className="topbar-right">
        {!running ? (
          <button
            className="button button-green"
            onClick={handleStart}
            disabled={startBusy}
          >
            {startBusy ? "Starting\u2026" : "Start Paper"}
          </button>
        ) : (
          <button
            className="button"
            onClick={handleStop}
            disabled={stopBusy}
          >
            {stopBusy ? "Stopping\u2026" : "Stop"}
          </button>
        )}
        <button
          className="button button-danger"
          onClick={handleFlatten}
          disabled={flattenBusy || !running}
        >
          {flattenBusy ? "Flattening\u2026" : "Flatten All"}
        </button>
      </div>
    </>
  );
}
