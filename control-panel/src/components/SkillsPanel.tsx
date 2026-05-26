import { useCallback, useEffect, useState } from "react";
import { Brain, Download, RefreshCw, Sparkles, TrendingUp } from "lucide-react";
import { api, API_BASE, type Skill } from "../lib/api";

function fmtDate(value: string | null): string {
  if (!value) return "never";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function ratePct(rate: number): string {
  return `${Math.round(rate * 100)}%`;
}

export function SkillsPanel() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.get<Skill[]>("/api/skills");
      setSkills(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <Brain className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Skills</h2>
        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
          {skills.length}
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
        Distilled and hand-authored skills. Version &gt; 1 means a skill has been
        self-improved (evidence-gated; regressions auto-roll-back).
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}
      {loading && <div className="text-sm text-zinc-500">Loading…</div>}
      {!loading && !error && skills.length === 0 && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-8 text-center text-sm text-zinc-500">
          No skills yet. They appear here as the agent distills successful tasks or
          authors them with the auto skill maker.
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2">
        {skills.map((skill) => (
          <div
            key={skill.name}
            className="lift flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-900 p-4"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={15} className="text-violet-400" />
              <span className="truncate font-mono text-sm font-semibold text-zinc-100">
                {skill.name}
              </span>
              {skill.version > 1 && (
                <span
                  title="Self-improved"
                  className="flex items-center gap-1 rounded-full bg-violet-500/15 px-2 py-0.5 text-[11px] font-medium text-violet-300"
                >
                  <TrendingUp size={11} /> v{skill.version}
                </span>
              )}
              <a
                href={`${API_BASE}/api/skills/${encodeURIComponent(skill.name)}/export.md`}
                target="_blank"
                rel="noreferrer"
                title="Export as agentskills.io SKILL.md"
                className="ml-auto flex size-7 items-center justify-center rounded-md text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <Download size={14} />
              </a>
            </div>

            {skill.description && (
              <p className="line-clamp-2 text-xs text-zinc-400">{skill.description}</p>
            )}

            {skill.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {skill.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}

            <div className="mt-1 flex items-center gap-4 text-xs text-zinc-500">
              <span title="Total invocations">{skill.use_count} uses</span>
              <span
                className={
                  skill.success_rate >= 0.8
                    ? "text-emerald-400"
                    : skill.success_rate >= 0.5
                      ? "text-amber-400"
                      : "text-red-400"
                }
                title="Measured success rate"
              >
                {ratePct(skill.success_rate)} success
              </span>
              <span className="ml-auto">used {fmtDate(skill.last_used)}</span>
            </div>

            <div className="h-1 overflow-hidden rounded-full bg-zinc-800" title="Success rate">
              <div
                className={`bar-grow h-full rounded-full ${
                  skill.success_rate >= 0.8
                    ? "bg-emerald-500"
                    : skill.success_rate >= 0.5
                      ? "bg-amber-500"
                      : "bg-red-500"
                }`}
                style={{ width: `${Math.round(skill.success_rate * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
