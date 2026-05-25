import { useEffect, useState } from "react";
import { Activity, Brain, CalendarClock, MessageSquare, Settings } from "lucide-react";
import { api, type Health } from "../lib/api";
import { OverviewPanel } from "./OverviewPanel";
import { ChatInterface } from "./ChatInterface";
import { SkillsPanel } from "./SkillsPanel";
import { SchedulePanel } from "./SchedulePanel";
import { SettingsPanel } from "./SettingsPanel";

type Section = "overview" | "chat" | "skills" | "schedule" | "settings";

const NAV: { id: Section; label: string; icon: typeof MessageSquare }[] = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "skills", label: "Skills", icon: Brain },
  { id: "schedule", label: "Schedule", icon: CalendarClock },
  { id: "settings", label: "Settings", icon: Settings },
];

function useHealth(): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        const h = await api.get<Health>("/health");
        if (alive) setOk(h.status === "ok");
      } catch {
        if (alive) setOk(false);
      }
    };
    void ping();
    const id = setInterval(() => void ping(), 10_000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return ok;
}

export function Dashboard() {
  const [section, setSection] = useState<Section>("overview");
  const health = useHealth();

  return (
    <div className="flex h-full min-h-0">
      {/* Left navigation rail */}
      <nav className="flex w-16 shrink-0 flex-col items-center gap-1 border-r border-zinc-800 bg-zinc-950 py-3 sm:w-44 sm:items-stretch sm:px-2">
        <div className="mb-3 flex items-center gap-2 px-2 sm:px-2">
          <span className="flex size-9 items-center justify-center rounded-lg bg-violet-600 text-sm font-bold text-white">
            AI
          </span>
          <span className="hidden text-sm font-semibold text-zinc-200 sm:inline">agent-ai</span>
        </div>

        {NAV.map(({ id, label, icon: Icon }) => {
          const active = section === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setSection(id)}
              title={label}
              className={`flex items-center justify-center gap-2.5 rounded-lg px-2 py-2.5 text-sm transition-colors sm:justify-start ${
                active
                  ? "bg-zinc-800 text-white"
                  : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
              }`}
            >
              <Icon size={18} className="shrink-0" />
              <span className="hidden sm:inline">{label}</span>
            </button>
          );
        })}

        <div className="mt-auto flex items-center gap-2 px-2 py-2" title="Gateway health">
          <span
            className={`inline-block size-2 rounded-full ${
              health === null
                ? "bg-amber-400 animate-pulse"
                : health
                  ? "bg-emerald-400"
                  : "bg-red-400"
            }`}
          />
          <span className="hidden text-xs text-zinc-500 sm:inline">
            {health === null ? "checking" : health ? "online" : "offline"}
          </span>
        </div>
      </nav>

      {/* Section content */}
      <div className="min-w-0 flex-1 overflow-hidden">
        {section === "chat" ? (
          <ChatInterface />
        ) : (
          <div className="h-full overflow-y-auto bg-zinc-950 px-4 py-6 sm:px-8">
            {section === "overview" && <OverviewPanel />}
            {section === "skills" && <SkillsPanel />}
            {section === "schedule" && <SchedulePanel />}
            {section === "settings" && <SettingsPanel />}
          </div>
        )}
      </div>
    </div>
  );
}
