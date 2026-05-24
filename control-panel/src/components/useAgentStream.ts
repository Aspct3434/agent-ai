import { useCallback, useEffect, useRef, useState } from "react";

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
  /** "iteration_limit" - hit MAX_REACT_ITERATIONS; "exception" - unhandled Python error */
  reason:
    | "iteration_limit"
    | "exception"
    | "rate_limited"
    | "critical_failure"
    | "cancelled";
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
  sendMessage: (text: string, sessionId?: string) => void;
  stopMessage: (sessionId?: string) => void;
  clearEvents: () => void;
}

const BASE_DELAY_MS = 1_000;
const MAX_DELAY_MS = 30_000;

export function useAgentStream(url: string): UseAgentStreamReturn {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [streamingText, setStreamingText] = useState<string>("");
  const [status, setStatus] = useState<ConnectionStatus>("connecting");

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(BASE_DELAY_MS);
  const sessionIdRef = useRef(crypto.randomUUID());
  const connectRef = useRef<() => void>(() => {});
  const activeRef = useRef(false);

  useEffect(() => {
    activeRef.current = true;
    reconnectDelayRef.current = BASE_DELAY_MS;

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

      ws.onmessage = (event) => {
        if (!activeRef.current) return;
        let payload: unknown;
        try {
          payload = JSON.parse(event.data as string);
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

        const parsed = payload as { type: string };
        if (parsed.type === "token") {
          setStreamingText((previous) => previous + (parsed as TokenEvent).content);
        } else if (
          parsed.type === "tool_call" ||
          parsed.type === "text" ||
          parsed.type === "final_answer"
        ) {
          if (parsed.type === "text" || parsed.type === "final_answer") {
            setStreamingText("");
          }
          setEvents((previous) => [...previous, parsed as AgentEvent]);
        }
      };

      ws.onerror = () => {
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

  const sendMessage = useCallback((text: string, sessionId?: string) => {
    const ws = socketRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ session_id: sessionId ?? sessionIdRef.current, text }));
  }, []);

  const stopMessage = useCallback((sessionId?: string) => {
    const targetSessionId = sessionId ?? sessionIdRef.current;
    const ws = socketRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cancel", session_id: targetSessionId }));
    }
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setStreamingText("");
  }, []);

  return { events, streamingText, status, sendMessage, stopMessage, clearEvents };
}
