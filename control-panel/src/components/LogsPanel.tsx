import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, ScrollText, Trash2 } from "lucide-react";
import { api, type LogEntry } from "../lib/api";

const LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"] as const;

const LEVEL_COLOR: Record<string, string> = {
  DEBUG: "text-zinc-500",
  INFO: "text-zinc-300",
  WARNING: "text-amber-400",
  ERROR: "text-red-400",
  CRITICAL: "text-red-400",
};

function fmtTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toLocaleTimeString(undefined, { hour12: false }) + "." +
    String(d.getMilliseconds()).padStart(3, "0");
}

export function LogsPanel() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [level, setLevel] = useState<(typeof LEVELS)[number]>("ALL");
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async (lvl: string) => {
    try {
      const data = await api.get<LogEntry[]>(`/api/logs?level=${lvl}&limit=300`);
      setLogs(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional poll-on-mount
    void load(level);
    if (paused) return;
    const id = setInterval(() => void load(level), 2000);
    return () => clearInterval(id);
  }, [load, level, paused]);

  useEffect(() => {
    if (!paused) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, paused]);

  async function clear() {
    try {
      await api.del("/api/logs");
      await load(level);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-5xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <ScrollText className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Logs</h2>
        {!paused && (
          <span className="flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-300">
            <span className="inline-block size-1.5 rounded-full bg-emerald-400 [animation:agentai-pulse_2s_ease-in-out_infinite]" />
            live
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value as typeof level)}
            className="rounded-lg border border-zinc-700 bg-zinc-800 px-2 py-1.5 text-xs text-zinc-100 outline-none focus:border-violet-600"
          >
            {LEVELS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setPaused((p) => !p)}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
          >
            {paused ? <Play size={13} /> : <Pause size={13} />}
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            type="button"
            onClick={() => void clear()}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:border-red-500/50 hover:text-red-300"
          >
            <Trash2 size={13} /> Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs leading-relaxed">
        {logs.length === 0 ? (
          <div className="py-8 text-center text-zinc-600">No log records yet.</div>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="flex gap-2 whitespace-pre-wrap break-words">
              <span className="shrink-0 text-zinc-600">{fmtTime(log.time)}</span>
              <span className={`w-16 shrink-0 ${LEVEL_COLOR[log.level] ?? "text-zinc-400"}`}>
                {log.level}
              </span>
              <span className="shrink-0 text-violet-400/70">{log.name}</span>
              <span className={LEVEL_COLOR[log.level] ?? "text-zinc-300"}>{log.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
