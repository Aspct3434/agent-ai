import { useCallback, useEffect, useState } from "react";
import { HardDrive, RefreshCw, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import { KeyValues } from "./KeyValues";

export function MemoryPanel() {
  const [profile, setProfile] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.get<Record<string, unknown>>("/api/profile");
      setProfile(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

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
    </div>
  );
}
