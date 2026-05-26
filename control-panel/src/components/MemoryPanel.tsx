import { useCallback, useEffect, useState } from "react";
import { HardDrive, RefreshCw, Search, Trash2 } from "lucide-react";
import { api, type SessionMatch } from "../lib/api";
import { KeyValues } from "./KeyValues";

function fmtTime(ts: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(ts * 1000));
}

export function MemoryPanel() {
  const [profile, setProfile] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<SessionMatch[] | null>(null);
  const [searching, setSearching] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.get<Record<string, unknown>>("/api/profile");
      setProfile(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) {
      setMatches(null);
      return;
    }
    setSearching(true);
    try {
      const res = await api.get<SessionMatch[]>(
        `/api/sessions/search?q=${encodeURIComponent(q)}&limit=25`,
      );
      setMatches(res);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  async function clear() {
    try {
      await api.del("/api/profile");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <HardDrive className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Memory</h2>
        <button
          type="button"
          onClick={() => void load()}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
        >
          <RefreshCw size={13} /> Refresh
        </button>
        <button
          type="button"
          onClick={() => void clear()}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:border-red-500/50 hover:text-red-300"
        >
          <Trash2 size={13} /> Clear
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        The agent's learned model of you — extracted across conversations and injected
        into each new session for personalization.
      </p>
      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}
      <KeyValues data={profile} />

      <section className="flex flex-col gap-3 pt-2">
        <h3 className="text-sm font-semibold text-zinc-300">Recall past conversations</h3>
        <form onSubmit={search} className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search
              size={14}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500"
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search everything the agent has discussed…"
              className="w-full rounded-lg border border-zinc-700 bg-zinc-800 py-2 pl-9 pr-3 text-sm text-zinc-100 outline-none focus:border-violet-600"
            />
          </div>
          <button
            type="submit"
            disabled={searching}
            className="rounded-lg bg-violet-700 px-4 py-2 text-xs text-white hover:bg-violet-600 disabled:opacity-50"
          >
            {searching ? "…" : "Search"}
          </button>
        </form>

        {matches !== null && matches.length === 0 && (
          <p className="text-sm text-zinc-500">No matches.</p>
        )}
        <div className="flex flex-col gap-2">
          {matches?.map((m, i) => (
            <div key={i} className="lift rounded-lg border border-zinc-800 bg-zinc-900 p-3">
              <div className="mb-1 flex items-center gap-2 text-[11px] text-zinc-500">
                <span
                  className={`rounded px-1.5 py-0.5 ${
                    m.role === "user" ? "bg-violet-500/15 text-violet-300" : "bg-zinc-800 text-zinc-400"
                  }`}
                >
                  {m.role}
                </span>
                <span className="truncate font-mono">{m.session_id}</span>
                <span className="ml-auto">{fmtTime(m.ts)}</span>
              </div>
              <p className="line-clamp-3 text-sm text-zinc-300">{m.snippet || m.content}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
