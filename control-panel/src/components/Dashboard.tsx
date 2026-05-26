import { useEffect, useState } from "react";
import {
  Activity,
  Brain,
  CalendarClock,
  HardDrive,
  MessageSquare,
  Settings,
  User,
  Wrench,
} from "lucide-react";
import { api, type Health } from "../lib/api";
import { OverviewPanel } from "./OverviewPanel";
import { ChatInterface } from "./ChatInterface";
import { SkillsPanel } from "./SkillsPanel";
import { SchedulePanel } from "./SchedulePanel";
import { SettingsPanel } from "./SettingsPanel";
import { MemoryPanel } from "./MemoryPanel";
import { PersonaPanel } from "./PersonaPanel";

type Section =
  | "overview"
  | "chat"
  | "skills"
  | "schedule"
  | "tools"
  | "memory"
  | "persona"
  | "logs"
  | "settings";

interface NavSpec {
  id: Section;
  label: string;
  icon: typeof MessageSquare;
  dot?: boolean;
}

const NAV_CORE: NavSpec[] = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "skills", label: "Skills", icon: Brain },
  { id: "schedule", label: "Schedule", icon: CalendarClock },
];

const NAV_EXTENDED: NavSpec[] = [
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "memory", label: "Memory", icon: HardDrive },
  { id: "persona", label: "Persona", icon: User },
  { id: "logs", label: "Logs", icon: Activity },
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

function NavItem({
  item,
  active,
  onClick,
}: {
  item: NavSpec;
  active: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      onClick={onClick}
      title={item.label}
      className={`flex items-center justify-center gap-2.5 rounded-lg px-2.5 py-2.5 text-sm transition-colors sm:justify-start ${
        active
          ? "bg-zinc-800 text-white"
          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
      }`}
    >
      <Icon size={18} className="shrink-0" />
      <span className="hidden flex-1 text-left sm:inline">{item.label}</span>
      {item.dot && (
        <span className="hidden size-1.5 rounded-full bg-red-500 sm:inline-block" />
      )}
    </button>
  );
}

function NavGroup({ label }: { label: string }) {
  return (
    <div className="hidden px-2.5 pb-1 pt-3 text-[9.5px] font-medium uppercase tracking-[0.08em] text-zinc-600 sm:block">
      {label}
    </div>
  );
}

function PlaceholderPanel({
  icon: Icon,
  title,
  blurb,
}: {
  icon: typeof MessageSquare;
  title: string;
  blurb: string;
}) {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <Icon className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">{title}</h2>
      </div>
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-6 py-10 text-center text-sm leading-relaxed text-zinc-500">
        {blurb}
      </div>
    </div>
  );
}

export function Dashboard() {
  const [section, setSection] = useState<Section>("overview");
  const health = useHealth();

  const renderItem = (item: NavSpec) => (
    <NavItem
      key={item.id}
      item={item}
      active={section === item.id}
      onClick={() => setSection(item.id)}
    />
  );

  return (
    <div className="flex h-full min-h-0">
      {/* Left navigation rail */}
      <nav className="flex w-16 shrink-0 flex-col gap-0.5 overflow-y-auto border-r border-zinc-800 bg-zinc-950 px-2 py-3 sm:w-[188px]">
        <div className="mb-2 flex items-center gap-2 px-1.5 pb-2">
          <span className="flex size-9 items-center justify-center rounded-lg bg-violet-700 text-sm font-bold text-white">
            AI
          </span>
          <span className="font-display hidden text-base text-zinc-200 sm:inline">
            agent-ai
          </span>
        </div>

        <NavGroup label="Core" />
        {NAV_CORE.map(renderItem)}
        <NavGroup label="Extended" />
        {NAV_EXTENDED.map(renderItem)}

        <div
          className="mt-auto flex items-center gap-2 border-t border-zinc-800 px-2.5 pb-1 pt-3"
          title="Gateway health"
        >
          <span
            className={`inline-block size-2 rounded-full ${
              health === null
                ? "bg-amber-400 [animation:agentai-pulse_2s_ease-in-out_infinite]"
                : health
                  ? "bg-emerald-500"
                  : "bg-red-500"
            }`}
          />
          <span className="hidden text-xs text-zinc-500 sm:inline">
            {health === null ? "connecting" : health ? "online" : "offline"}
          </span>
          <span className="ml-auto hidden font-mono text-[10px] text-zinc-600 sm:inline">
            v0.1.6
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
            {section === "memory" && <MemoryPanel />}
            {section === "persona" && <PersonaPanel />}
            {section === "settings" && <SettingsPanel />}
            {section === "tools" && (
              <PlaceholderPanel
                icon={Wrench}
                title="Tools"
                blurb="MCP servers, terminal, file, process, and port tools. Inspect, enable, and configure each adapter. (Live tool inventory coming soon.)"
              />
            )}
            {section === "logs" && (
              <PlaceholderPanel
                icon={Activity}
                title="Logs"
                blurb="Live gateway logs, per-session tool traces, and error replays. (Streaming log viewer coming soon.)"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
