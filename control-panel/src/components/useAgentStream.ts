import { useCallback, useEffect, useRef, useState } from "react";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ToolCallEvent = {
  type: "tool_call";
  tool: string;
  params: Record<string, unknown>;
};

export type TextEvent = {
  type: "text";
  content: string;
};

export type FinalAnswerEvent = {
  type: "final_answer";
  /** "iteration_limit" — hit MAX_REACT_ITERATIONS; "exception" — unhandled Python error */
  reason: "iteration_limit" | "exception" | "rate_limited" | "critical_failure";
  content: string;
};

export type TokenEvent = {
  type: "token";
  content: string;
};

export type AgentEvent = ToolCallEvent | TextEvent | FinalAnswerEvent;

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export interface UseAgentStreamReturn {
  events: AgentEvent[];
  streamingText: string;
  status: ConnectionStatus;
  sendMessage: (text: string) => void;
  clearEvents: () => void;
}

// ---------------------------------------------------------------------------
// Reconnect constants
// ---------------------------------------------------------------------------

const BASE_DELAY_MS = 1_000;
const MAX_DELAY_MS = 30_000;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAgentStream(url: string): UseAgentStreamReturn {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [streamingText, setStreamingText] = useState<string>("");
  const [status, setStatus] = useState<ConnectionStatus>("connecting");

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(BASE_DELAY_MS);
  // Stable session ID for the lifetime of this hook instance
  const sessionIdRef = useRef(crypto.randomUUID());
  // Always points to the latest connect() closure so onclose can call it
  const connectRef = useRef<() => void>(() => {});
  // Prevents state updates after unmount or during url-change teardown
  const activeRef = useRef(false);

  useEffect(() => {
    activeRef.current = true;
    reconnectDelayRef.current = BASE_DELAY_MS;
    setEvents([]);

    function connect() {
      if (!activeRef.current) return;
      setStatus("connecting");

      const ws = new WebSocket(url);
      socketRef.current = ws;

      ws.onopen = () => {
        if (!activeRef.current) return;
        reconnectDelayRef.current = BASE_DELAY_MS;
        setStatus("connected");
      };

      ws.onmessage = (ev) => {
        if (!activeRef.current) return;
        let payload: unknown;
        try {
          payload = JSON.parse(ev.data as string);
        } catch {
          return;
        }
        if (
          typeof payload !== "object" ||
          payload === null ||
          !("type" in payload)
        ) {
          return;
        }
        const p = payload as { type: string };
        if (p.type === "token") {
          setStreamingText((prev) => prev + (p as TokenEvent).content);
        } else if (
          p.type === "tool_call" ||
          p.type === "text" ||
          p.type === "final_answer"
        ) {
          if (p.type === "text" || p.type === "final_answer") {
            setStreamingText("");
          }
          setEvents((prev) => [...prev, p as AgentEvent]);
        }
      };

      ws.onerror = () => {
        // onclose always fires after onerror, so just let onclose drive reconnect
        ws.close();
      };

      ws.onclose = () => {
        if (!activeRef.current) return;
        setStatus("disconnected");
        reconnectTimerRef.current = setTimeout(() => {
          reconnectDelayRef.current = Math.min(
            reconnectDelayRef.current * 2,
            MAX_DELAY_MS,
          );
          connectRef.current();
        }, reconnectDelayRef.current);
      };
    }

    // Keep the ref current so the onclose closure above always calls the right
    // version even after the url changes and this effect re-runs
    connectRef.current = connect;
    connect();

    return () => {
      activeRef.current = false;
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [url]);

  const sendMessage = useCallback((text: string) => {
    const ws = socketRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ session_id: sessionIdRef.current, text }));
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setStreamingText("");
  }, []);

  return { events, streamingText, status, sendMessage, clearEvents };
}
