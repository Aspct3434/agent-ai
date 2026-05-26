import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { AlertTriangle, CheckCircle2, MessageSquare, Plus, Send, Square, Trash2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAgentStream, type AgentEvent } from "./useAgentStream";
import { ToolBadge } from "./ToolBadge";

const WS_URL = "ws://127.0.0.1:8000/ws/stream";
const HISTORY_STORAGE_KEY = "agent-control-panel.chat-history.v1";
const ACTIVE_CHAT_STORAGE_KEY = "agent-control-panel.active-chat.v1";
const MAX_SAVED_CONVERSATIONS = 40;

interface Turn {
  id: string;
  userText: string;
  events: AgentEvent[];
  createdAt: number;
  completedAt: number;
}

interface Conversation {
  id: string;
  sessionId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: Turn[];
}

function createConversation(): Conversation {
  const now = Date.now();
  const id = crypto.randomUUID();
  return {
    id,
    sessionId: crypto.randomUUID(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

function titleFromMessage(text: string): string {
  const title = text.replace(/\s+/g, " ").trim();
  return title.length > 46 ? `${title.slice(0, 43).trim()}...` : title || "New chat";
}

function formatTimestamp(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(value);
}

function sortConversations(conversations: Conversation[]): Conversation[] {
  return [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_SAVED_CONVERSATIONS);
}

function isConversation(value: unknown): value is Conversation {
  if (typeof value !== "object" || value === null) return false;
  const item = value as Partial<Conversation>;
  return (
    typeof item.id === "string" &&
    typeof item.sessionId === "string" &&
    typeof item.title === "string" &&
    typeof item.createdAt === "number" &&
    typeof item.updatedAt === "number" &&
    Array.isArray(item.turns)
  );
}

function loadHistoryState(): { conversations: Conversation[]; activeId: string } {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    const loaded = Array.isArray(parsed)
      ? sortConversations(parsed.filter(isConversation))
      : [];
    if (loaded.length > 0) {
      const savedActive = localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
      return {
        conversations: loaded,
        activeId: loaded.some((conversation) => conversation.id === savedActive)
          ? String(savedActive)
          : loaded[0].id,
      };
    }
  } catch {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
    localStorage.removeItem(ACTIVE_CHAT_STORAGE_KEY);
  }

  const firstConversation = createConversation();
  return { conversations: [firstConversation], activeId: firstConversation.id };
}

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
      <p className="max-w-[78%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-violet-700 px-4 py-2.5 text-sm text-white">
        {text}
      </p>
    </div>
  );
}

function AgentTextBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-start">
      <div className="agent-markdown max-w-[78%] rounded-2xl rounded-bl-md bg-zinc-800 px-4 py-2.5 text-sm leading-relaxed text-zinc-100">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

function StreamingTextBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-start">
      <div className="agent-markdown max-w-[78%] rounded-2xl rounded-bl-md bg-zinc-800 px-4 py-2.5 text-sm leading-relaxed text-zinc-100">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-zinc-400 align-middle" />
      </div>
    </div>
  );
}

function ToolResultBubble({
  toolName,
  isError,
  content,
}: {
  toolName: string;
  isError: boolean;
  content: string;
}) {
  const Icon = isError ? AlertTriangle : CheckCircle2;
  const tone = isError
    ? "border-red-500/40 bg-red-950/30 text-red-100"
    : "border-emerald-500/30 bg-emerald-950/20 text-emerald-100";
  return (
    <div className="flex justify-start">
      <div className={`max-w-[86%] overflow-hidden rounded-lg border ${tone}`}>
        <div className="flex items-center gap-2 border-b border-white/10 px-3 py-2 text-xs font-semibold">
          <Icon size={14} className="shrink-0" />
          <span className="truncate">{toolName}</span>
          <span className="ml-auto shrink-0 font-mono text-[11px] uppercase opacity-70">
            {isError ? "error" : "result"}
          </span>
        </div>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words px-3 py-2 font-mono text-xs leading-relaxed">
          {content}
        </pre>
      </div>
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-md bg-zinc-800 px-4 py-3.5">
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
  if (event.type === "tool_result") {
    return (
      <ToolResultBubble
        toolName={event.tool}
        isError={event.is_error}
        content={event.content}
      />
    );
  }
  return <AgentTextBubble content={event.content} />;
}

function TurnBlock({ turn }: { turn: Turn }) {
  return (
    <div className="flex flex-col gap-3">
      <UserBubble text={turn.userText} />
      {turn.events.map((event, index) => (
        <EventRow key={`${turn.id}-${index}`} event={event} />
      ))}
    </div>
  );
}

export function ChatInterface() {
  const { events, streamingText, status, sendMessage, stopMessage, clearEvents } =
    useAgentStream(WS_URL);
  const [initialState] = useState(loadHistoryState);
  const [conversations, setConversations] = useState<Conversation[]>(
    initialState.conversations,
  );
  const [activeConversationId, setActiveConversationId] = useState(initialState.activeId);
  const [activeUserText, setActiveUserText] = useState<string | null>(null);
  const [input, setInput] = useState("");

  const bottomRef = useRef<HTMLDivElement>(null);
  const liveTurnStartedAtRef = useRef<number | null>(null);

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0];

  const lastEvent = events.length > 0 ? events[events.length - 1] : null;
  const executionStatus: "idle" | "thinking" | "executing_tool" | "success" =
    activeUserText === null
      ? "idle"
      : lastEvent?.type === "text" || lastEvent?.type === "final_answer"
        ? "success"
        : lastEvent?.type === "tool_call"
          ? "executing_tool"
          : "thinking";

  const isAgentBusy =
    executionStatus === "thinking" || executionStatus === "executing_tool";
  const isStreaming = isAgentBusy;
  const canSend = status === "connected" && !isAgentBusy && input.trim().length > 0;
  const canNavigateHistory = !isAgentBusy;

  const commitLiveTurn = useCallback(() => {
    if (activeUserText === null || !activeConversation) return;

    const now = Date.now();
    const turn: Turn = {
      id: crypto.randomUUID(),
      userText: activeUserText,
      events: [...events],
      createdAt: liveTurnStartedAtRef.current ?? now,
      completedAt: now,
    };

    setConversations((previous) =>
      sortConversations(
        previous.map((conversation) =>
          conversation.id === activeConversation.id
            ? {
                ...conversation,
                title:
                  conversation.turns.length === 0
                    ? titleFromMessage(activeUserText)
                    : conversation.title,
                turns: [...conversation.turns, turn],
                updatedAt: now,
              }
            : conversation,
        ),
      ),
    );
    setActiveUserText(null);
    liveTurnStartedAtRef.current = null;
    clearEvents();
  }, [activeConversation, activeUserText, clearEvents, events]);

  useEffect(() => {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(conversations));
    localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, activeConversationId);
  }, [activeConversationId, conversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeConversationId, activeUserText, conversations, events, streamingText]);

  useEffect(() => {
    if (activeUserText !== null && executionStatus === "success") {
      commitLiveTurn();
    }
  }, [activeUserText, commitLiveTurn, executionStatus]);

  function handleNewChat() {
    if (!canNavigateHistory) return;
    if (activeUserText !== null && executionStatus === "success") {
      commitLiveTurn();
    }

    const conversation = createConversation();
    setConversations((previous) => sortConversations([conversation, ...previous]));
    setActiveConversationId(conversation.id);
    setActiveUserText(null);
    liveTurnStartedAtRef.current = null;
    clearEvents();
  }

  function handleSelectConversation(conversationId: string) {
    if (!canNavigateHistory || conversationId === activeConversationId) return;
    if (activeUserText !== null && executionStatus === "success") {
      commitLiveTurn();
    }

    setActiveConversationId(conversationId);
    setActiveUserText(null);
    liveTurnStartedAtRef.current = null;
    clearEvents();
  }

  function handleDeleteConversation(conversationId: string) {
    if (isAgentBusy && conversationId === activeConversationId) return;

    setConversations((previous) => {
      const remaining = previous.filter((conversation) => conversation.id !== conversationId);
      if (remaining.length === 0) {
        const nextConversation = createConversation();
        setActiveConversationId(nextConversation.id);
        setActiveUserText(null);
        clearEvents();
        return [nextConversation];
      }

      if (conversationId === activeConversationId) {
        setActiveConversationId(remaining[0].id);
        setActiveUserText(null);
        clearEvents();
      }

      return remaining;
    });
  }

  function handleSubmit(e?: { preventDefault(): void }) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || status !== "connected" || isAgentBusy || !activeConversation) return;

    const now = Date.now();
    setConversations((previous) =>
      sortConversations(
        previous.map((conversation) =>
          conversation.id === activeConversation.id
            ? {
                ...conversation,
                title:
                  conversation.turns.length === 0
                    ? titleFromMessage(text)
                    : conversation.title,
                updatedAt: now,
              }
            : conversation,
        ),
      ),
    );

    clearEvents();
    liveTurnStartedAtRef.current = now;
    setActiveUserText(text);
    sendMessage(text, activeConversation.sessionId);
    setInput("");
  }

  function handleStop() {
    if (!activeConversation || !isAgentBusy) return;
    stopMessage(activeConversation.sessionId);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="flex h-full min-h-0 bg-zinc-950 text-zinc-100">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-zinc-800 bg-zinc-950 md:flex">
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-zinc-900 text-zinc-300">
            <MessageSquare size={17} />
          </span>
          <span className="text-sm font-semibold text-zinc-200">History</span>
          <button
            type="button"
            onClick={handleNewChat}
            disabled={!canNavigateHistory}
            aria-label="New chat"
            title="New chat"
            className="ml-auto flex size-11 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 text-zinc-200 transition-colors hover:border-violet-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Plus size={17} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {conversations.map((conversation) => {
            const isActive = conversation.id === activeConversationId;
            return (
              <div
                key={conversation.id}
                className={`group flex items-start gap-1 rounded-lg transition-colors ${
                  isActive
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
                }`}
              >
                <button
                  type="button"
                  onClick={() => handleSelectConversation(conversation.id)}
                  disabled={!canNavigateHistory && !isActive}
                  className="flex min-w-0 flex-1 items-start gap-2 rounded-lg px-3 py-2.5 text-left disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <MessageSquare className="mt-0.5 shrink-0" size={15} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">
                      {conversation.title}
                    </span>
                    <span className="mt-0.5 block truncate text-xs text-zinc-500">
                      {conversation.turns.length} turns - {formatTimestamp(conversation.updatedAt)}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  aria-label={`Delete ${conversation.title}`}
                  title="Delete chat"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDeleteConversation(conversation.id);
                  }}
                  className="mr-1 mt-1 flex size-9 shrink-0 items-center justify-center rounded-lg text-zinc-500 opacity-0 transition-opacity hover:bg-zinc-700 hover:text-red-300 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-zinc-950">
        <header className="flex items-center gap-2.5 border-b border-zinc-800 bg-zinc-900 px-4 py-3 shadow-sm">
          <StatusDot status={status} />
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-zinc-200">
              {activeConversation?.title ?? "Agent"}
            </div>
            <div className="font-mono text-xs capitalize text-zinc-500">{status}</div>
          </div>
          <select
            value={activeConversationId}
            onChange={(event) => handleSelectConversation(event.target.value)}
            disabled={!canNavigateHistory}
            aria-label="Chat history"
            className="ml-auto max-w-[45vw] rounded-lg border border-zinc-700 bg-zinc-800 px-2 py-2 text-xs text-zinc-100 outline-none focus:border-violet-600 disabled:opacity-50 md:hidden"
          >
            {conversations.map((conversation) => (
              <option key={conversation.id} value={conversation.id}>
                {conversation.title}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleNewChat}
            disabled={!canNavigateHistory}
            aria-label="New chat"
            title="New chat"
            className="flex size-11 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-800 text-zinc-200 transition-colors hover:border-violet-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-40 md:hidden"
          >
            <Plus size={17} />
          </button>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {activeConversation?.turns.length === 0 && activeUserText === null && (
              <div className="mt-24 text-center text-sm text-zinc-600">
                Send a message to start a conversation
              </div>
            )}

            {activeConversation?.turns.map((turn) => (
              <TurnBlock key={turn.id} turn={turn} />
            ))}

            {activeUserText !== null && (
              <div className="flex flex-col gap-3">
                <UserBubble text={activeUserText} />
                {events.map((event, index) => (
                  <EventRow key={`live-${index}`} event={event} />
                ))}
                {streamingText ? (
                  <StreamingTextBubble text={streamingText} />
                ) : isStreaming ? (
                  <ThinkingIndicator />
                ) : null}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </main>

        <footer className="border-t border-zinc-800 bg-zinc-900 px-4 pb-4 pt-3">
          <form
            onSubmit={handleSubmit}
            className="mx-auto flex max-w-3xl items-end gap-2"
          >
            <textarea
              rows={1}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isAgentBusy
                  ? "Agent is executing..."
                  : status === "connected"
                    ? "Message the agent..."
                    : "Waiting for connection..."
              }
              disabled={status !== "connected" || isAgentBusy}
              className="min-h-11 flex-1 resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-colors focus:border-violet-600 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <button
              type="button"
              onClick={handleStop}
              disabled={!isAgentBusy}
              aria-label="Stop agent task"
              title="Stop agent task"
              className="flex size-11 shrink-0 items-center justify-center rounded-lg border border-red-500/40 bg-red-500/10 text-red-300 transition-colors hover:border-red-400 hover:bg-red-500/20 hover:text-red-100 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <Square size={15} fill="currentColor" />
            </button>
            <button
              type="submit"
              disabled={!canSend}
              aria-label="Send message"
              className="flex size-11 shrink-0 items-center justify-center rounded-lg bg-violet-700 text-white transition-colors hover:bg-violet-600 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send size={17} />
            </button>
          </form>
          <p className="mt-2 text-center text-[11px] text-zinc-600">
            Enter to send - Shift+Enter for new line
          </p>
        </footer>
      </section>
    </div>
  );
}

export default ChatInterface;
