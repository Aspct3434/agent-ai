import { useCallback, useEffect, useState } from "react";
import { Check, RefreshCw, User } from "lucide-react";
import { api, type PersonaInfo } from "../lib/api";

export function PersonaPanel() {
  const [info, setInfo] = useState<PersonaInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.get<PersonaInfo>("/api/persona");
      setInfo(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  async function select(name: string) {
    setBusy(name);
    try {
      await api.post("/api/persona/load", { persona_name: name });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  const personas = info?.available_personas ?? [];

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <User className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Persona</h2>
        {info?.active_persona && (
          <span className="rounded-full bg-violet-500/15 px-2 py-0.5 text-[11px] font-medium text-violet-300">
            {info.active_persona}
          </span>
        )}
        <button
          type="button"
          onClick={() => void load()}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        The persona (SOUL.md / AGENTS.md) injected into the system prompt — the agent's
        identity, values, and tone. Selecting one hot-reloads it for new turns.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      {personas.length === 0 ? (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-6 text-sm text-zinc-400">
          {info?.loaded
            ? `Using the default persona (${info.content_length} chars from ${info.persona_dir ?? "persona/"}). Add named sub-folders under the persona directory to switch between personas here.`
            : "No persona loaded."}
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {personas.map((name) => {
            const active = info?.active_persona === name;
            return (
              <button
                key={name}
                type="button"
                onClick={() => void select(name)}
                disabled={busy !== null}
                className={`lift flex items-center gap-2 rounded-lg border px-4 py-3 text-left transition-colors disabled:opacity-50 ${
                  active
                    ? "border-violet-600 bg-violet-500/10"
                    : "border-zinc-800 bg-zinc-900 hover:border-zinc-700"
                }`}
              >
                <User size={16} className={active ? "text-violet-300" : "text-zinc-400"} />
                <span className={`font-medium ${active ? "text-violet-200" : "text-zinc-200"}`}>
                  {name}
                </span>
                {active && <Check size={15} className="ml-auto text-violet-300" />}
                {busy === name && <span className="ml-auto text-xs text-zinc-500">loading…</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
