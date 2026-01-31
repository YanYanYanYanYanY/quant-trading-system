import { useAppStore } from "../../store";

function ModeBadge() {
  const mode = useAppStore((s) => s.mode);
  const runState = useAppStore((s) => s.runState);
  return (
    <span className="badge">
      Mode: <b>{mode}</b> · <b>{runState}</b>
    </span>
  );
}

export function TopBar() {
  const canTrade = useAppStore((s) => s.canTrade);
  const killSwitchArmed = useAppStore((s) => s.killSwitchArmed);

  // These buttons are UI-only placeholders.
  // In your real app, you’ll call FastAPI endpoints here.
  return (
    <>
      <div className="row">
        <ModeBadge />
        <span className="badge">canTrade: <b>{String(canTrade)}</b></span>
        <span className="badge">killSwitch: <b>{killSwitchArmed ? "ARMED" : "off"}</b></span>
      </div>

      <div className="row">
        <button className="button">Start Paper</button>
        <button className="button">Pause</button>
        <button className="button danger">Flatten All</button>
      </div>
    </>
  );
}