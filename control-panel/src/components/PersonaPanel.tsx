import { useCallback, useEffect, useState } from "react";
import { RefreshCw, User } from "lucide-react";
import { api } from "../lib/api";
import { KeyValues } from "./KeyValues";

export function PersonaPanel() {
  const [persona, setPersona] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.get<Record<string, unknown>>("/api/persona");
      setPersona(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <User className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Persona</h2>
        <button
          type="button"
          onClick={() => void load()}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        The active persona (SOUL.md / AGENTS.md) injected into the system prompt — the
        agent's identity, values, and tone.
      </p>
      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}
      <KeyValues data={persona} />
    </div>
  );
}
