import {
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentStream, type AgentEvent } from "./useAgentStream";
import { ToolBadge } from "./ToolBadge";

const WS_URL = "ws://localhost:8000/ws/stream";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Turn {
  id: string;
  userText: string;
  events: AgentEvent[];
}

// ---------------------------------------------------------------------------
// Presentational sub-components
// ---------------------------------------------------------------------------

function StatusDot({ status }: { status: string }) {
  const color =
    status === "connected"
      ? "bg-emerald-400"
      : status === "disconnected"
        ? "bg-red-400"
        : "bg-amber-400 animate-pulse";
  return <span className={`inline-block size-2 rounded-full ${color}`} />;
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[70%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-violet-600 px-4 py-2.5 text-sm text-white">
        {text}
      </p>
    </div>
  );
}

function AgentTextBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-start">
      <div className="agent-markdown max-w-[70%] rounded-2xl rounded-bl-sm bg-zinc-800 px-4 py-2.5 text-sm leading-relaxed text-zinc-100">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

function StreamingTextBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-start">
      <div className="agent-markdown max-w-[70%] rounded-2xl rounded-bl-sm bg-zinc-800 px-4 py-2.5 text-sm leading-relaxed text-zinc-100">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-zinc-400 align-middle" />
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-zinc-800 px-4 py-3.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="size-1.5 animate-bounce rounded-full bg-zinc-400"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

function EventRow({ event }: { event: AgentEvent }) {
  if (event.type === "tool_call") {
    return <ToolBadge toolName={event.tool} params={event.params} />;
  }
  return <AgentTextBubble content={event.content} />;
}

// ---------------------------------------------------------------------------
// ChatInterface
// ---------------------------------------------------------------------------

export function ChatInterface() {
  const { events, streamingText, status, sendMessage, clearEvents } = useAgentStream(WS_URL);

  // Completed turns are archived here so past conversations stay visible
  const [pastTurns, setPastTurns] = useState<Turn[]>([]);
  // The user text of the turn currently being streamed
  const [activeUserText, setActiveUserText] = useState<string | null>(null);
  const [input, setInput] = useState("");

  const bottomRef = useRef<HTMLDivElement>(null);

  // Derive the agent's execution phase from the live event stream.
  // 'idle'           – no active turn
  // 'thinking'       – turn started, no events received yet
  // 'executing_tool' – last event is a tool_call (waiting for result / next step)
  // 'success'        – last event is text or final_answer (turn complete)
  const lastEvent = events.length > 0 ? events[events.length - 1] : null;
  const executionStatus: "idle" | "thinking" | "executing_tool" | "success" =
    activeUserText === null
      ? "idle"
      : events.length === 0
        ? "thinking"
        : lastEvent?.type === "tool_call"
          ? "executing_tool"
          : "success";

  const isAgentBusy =
    executionStatus === "thinking" || executionStatus === "executing_tool";
  const isStreaming = isAgentBusy;
  const canSend = status === "connected" && !isAgentBusy && input.trim().length > 0;

  // Keep the feed scrolled to the newest content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [pastTurns, events, streamingText]);

  function handleSubmit(e?: { preventDefault(): void }) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || status !== "connected" || isAgentBusy) return;

    // Archive the live turn before starting the next one
    if (activeUserText !== null) {
      setPastTurns((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          userText: activeUserText,
          events: [...events],
        },
      ]);
    }

    clearEvents();
    setActiveUserText(text);
    sendMessage(text);
    setInput("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950 text-zinc-100">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="flex items-center gap-2.5 border-b border-zinc-800 bg-zinc-900 px-4 py-3 shadow-sm">
        <StatusDot status={status} />
        <span className="text-sm font-semibold tracking-wide text-zinc-200">
          Agent
        </span>
        <span className="ml-auto font-mono text-xs capitalize text-zinc-500">
          {status}
        </span>
      </header>

      {/* ── Message feed ───────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-4">

          {/* Empty state */}
          {pastTurns.length === 0 && activeUserText === null && (
            <div className="mt-24 text-center text-sm text-zinc-600">
              Send a message to start a conversation
            </div>
          )}

          {/* Completed turns */}
          {pastTurns.map((turn) => (
            <div key={turn.id} className="flex flex-col gap-3">
              <UserBubble text={turn.userText} />
              {turn.events.map((ev, i) => (
                <EventRow key={i} event={ev} />
              ))}
            </div>
          ))}

          {/* Active turn — live stream */}
          {activeUserText !== null && (
            <div className="flex flex-col gap-3">
              <UserBubble text={activeUserText} />
              {events.map((ev, i) => (
                <EventRow key={i} event={ev} />
              ))}
              {streamingText ? (
                <StreamingTextBubble text={streamingText} />
              ) : isStreaming ? (
                <ThinkingIndicator />
              ) : null}
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── Input bar ──────────────────────────────────────────────────── */}
      <footer className="border-t border-zinc-800 bg-zinc-900 px-4 pb-4 pt-3">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-2xl items-end gap-2"
        >
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isAgentBusy
                ? "Agent is executing..."
                : status === "connected"
                  ? "Message the agent…"
                  : "Waiting for connection…"
            }
            disabled={status !== "connected" || isAgentBusy}
            className="flex-1 resize-none rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-colors focus:border-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!canSend}
            aria-label="Send message"
            className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-violet-600 text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send size={16} />
          </button>
        </form>
        <p className="mt-2 text-center text-[11px] text-zinc-600">
          Enter to send · Shift+Enter for new line
        </p>
      </footer>
    </div>
  );
}

export default ChatInterface;
