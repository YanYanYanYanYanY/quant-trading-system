import { create } from "zustand";
import type { AccountSnapshot, Order, Fill, Position, RiskMetrics, EngineMode, RunState, WSEvent } from "../app/types";

type ConnectionState = {
    apiUp: boolean;
    wsConnected: boolean;
    brokerAlive: boolean;
    marketDataAlive: boolean;
    lastHeartbeatTs?: number;
    setConnection: (p: Partial<ConnectionState>) => void;
};

type EngineState = {
    mode: EngineMode;
    runState: RunState;
    canTrade: boolean;
    warmupProgress: number; // 0..1
    killSwitchArmed: boolean;
    setEngine: (p: Partial<EngineState>) => void;
};

type AccountState = {
    snapshot?: AccountSnapshot;
    setAccount: (s: AccountSnapshot) => void;
};

type PositionsState = {
    bySymbol: Record<string, Position>;
    upsertPositions: (items: Position[]) => void;
};

type OrdersState = {
    byId: Record<string, Order>;
    ids: string[]; // newest first
    upsertOrders: (items: Order[]) => void;
};

type FillsState = {
    byId: Record<string, Fill>;
    byOrderId: Record<string, string[]>;
    upsertFills: (items: Fill[]) => void;
};

type RiskState = {
    metrics?: RiskMetrics;
    setRisk: (m: RiskMetrics) => void;
};

export type AppStore = ConnectionState &
    EngineState &
    AccountState &
    PositionsState &
    OrdersState &
    FillsState &
    RiskState & {
        applyEvent: (evt: WSEvent) => void;
    };

export const useAppStore = create<AppStore>((set, get) => ({
    // connection
    apiUp: true,
    wsConnected: false,
    brokerAlive: false,
    marketDataAlive: false,
    setConnection: (p) => set(p),

    // engine
    mode: "PAPER",
    runState: "STOPPED",
    canTrade: false,
    warmupProgress: 0,
    killSwitchArmed: false,
    setEngine: (p) => set(p),

    // account
    snapshot: undefined,
    setAccount: (s) => set({ snapshot: s }),

    // positions
    bySymbol: {},
    upsertPositions: (items) =>
        set((st) => {
            const next = { ...st.bySymbol };
            for (const it of items) next[it.symbol] = it;
            return { bySymbol: next };
        }),

    // orders
    byId: {},
    ids: [],
    upsertOrders: (items) =>
        set((st) => {
            const byId = { ...st.byId };
            const ids = [...st.ids];
            for (const o of items) {
                const isNew = !byId[o.id];
                byId[o.id] = o;
                if (isNew) ids.unshift(o.id);
            }
            return { byId, ids: ids.slice(0, 5000) };
        }),

    // fills
    byId: {},
    byOrderId: {},
    upsertFills: (items) =>
        set((st) => {
            const byId = { ...st.byId };
            const byOrderId = { ...st.byOrderId };
            for (const f of items) {
                const isNew = !byId[f.id];
                byId[f.id] = f;
                if (isNew) {
                    const arr = byOrderId[f.orderId] ? [...byOrderId[f.orderId]] : [];
                    arr.unshift(f.id);
                    byOrderId[f.orderId] = arr.slice(0, 200);
                }
            }
            return { byId, byOrderId };
        }),

    // risk
    metrics: undefined,
    setRisk: (m) => set({ metrics: m }),

    // websocket event router
    applyEvent: (evt) => {
        const s = get();
        switch (evt.type) {
            case "HEARTBEAT":
                s.setConnection({ lastHeartbeatTs: Date.now() });
                break;
            case "CONNECTION":
                s.setConnection(evt.payload);
                break;
            case "ENGINE_STATE":
                s.setEngine(evt.payload);
                break;
            case "ACCOUNT_SNAPSHOT":
                s.setAccount(evt.payload);
                break;
            case "POSITIONS":
                s.upsertPositions(evt.payload);
                break;
            case "ORDERS":
                s.upsertOrders(evt.payload);
                break;
            case "FILLS":
                s.upsertFills(evt.payload);
                break;
            case "RISK_METRICS":
                s.setRisk(evt.payload);
                break;
            default:
                // ignore unknown events for now
                break;
        }
    },
}));