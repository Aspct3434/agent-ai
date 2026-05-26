import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw, Server, Wrench } from "lucide-react";
import { api, type Tool } from "../lib/api";
import { useCountUp } from "../lib/useCountUp";

const SERVER_LABEL: Record<string, string> = {
  __builtin__: "Built-in",
  skills: "Skills",
  sqlite: "SQLite",
  filesystem: "Filesystem",
};

export function ToolsPanel() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [error, setError] = useState<string | null>(null);
  const total = useCountUp(tools.length);

  const load = useCallback(async () => {
    try {
      const data = await api.get<Tool[]>("/api/tools");
      setTools(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  const grouped = useMemo(() => {
    const by: Record<string, Tool[]> = {};
    for (const t of tools) (by[t.server] ??= []).push(t);
    return Object.entries(by).sort((a, b) => b[1].length - a[1].length);
  }, [tools]);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <Wrench className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Tools</h2>
        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs tabular-nums text-zinc-400">
          {total}
        </span>
        <button
          type="button"
          onClick={() => void load()}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        Every tool the agent can call — built-ins plus each connected MCP server.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      {grouped.map(([server, serverTools]) => (
        <section key={server} className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <Server size={15} className="text-violet-400" />
            {SERVER_LABEL[server] ?? server}
            <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400">
              {serverTools.length}
            </span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {serverTools.map((tool) => (
              <div
                key={`${server}:${tool.name}`}
                className="lift flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-900 p-3"
              >
                <span className="font-mono text-sm text-zinc-100">{tool.name}</span>
                {tool.description && (
                  <span className="line-clamp-2 text-xs text-zinc-500">{tool.description}</span>
                )}
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
