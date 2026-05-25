import { useEffect, useState } from "react";
import {
  Activity,
  Brain,
  CalendarClock,
  Cpu,
  HardDrive,
  MessageCircle,
  Radio,
  Send,
} from "lucide-react";
import { api, type Status } from "../lib/api";

const CHANNEL_META: { key: keyof Status["channels"]; label: string; icon: typeof Send }[] = [
  { key: "telegram", label: "Telegram", icon: Send },
  { key: "discord", label: "Discord", icon: MessageCircle },
  { key: "slack", label: "Slack", icon: Radio },
];

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: typeof Brain;
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex items-center gap-2 text-zinc-400">
        <Icon size={16} className="text-violet-400" />
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <div className="text-2xl font-semibold text-zinc-100">{value}</div>
      {sub && <div className="text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

export function OverviewPanel() {
  const [status, setStatus] = useState<Status | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await api.get<Status>("/api/status");
        if (alive) {
          setStatus(s);
          setOnline(true);
        }
      } catch {
        if (alive) setOnline(false);
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 5000); // mirror Hermes' 5s refresh
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5">
      <div className="flex items-center gap-3">
        <Activity className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Overview</h2>
        <span
          className={`ml-auto flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
            online === null
              ? "bg-amber-500/15 text-amber-300"
              : online
                ? "bg-emerald-500/15 text-emerald-300"
                : "bg-red-500/15 text-red-300"
          }`}
        >
          <span
            className={`inline-block size-2 rounded-full ${
              online === null
                ? "bg-amber-400 animate-pulse"
                : online
                  ? "bg-emerald-400"
                  : "bg-red-400"
            }`}
          />
          {online === null ? "connecting" : online ? "gateway online" : "gateway offline"}
        </span>
      </div>

      {online === false && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-4 py-3 text-sm text-red-200">
          Can't reach the gateway at <span className="font-mono">127.0.0.1:8000</span>. Start it
          with <span className="font-mono">uvicorn gateway:app --app-dir src</span>.
        </div>
      )}

      {/* Stat cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={Brain}
          label="Skills"
          value={status?.skills.count ?? "—"}
          sub={status ? `${status.skills.improved} self-improved` : undefined}
        />
        <StatCard
          icon={CalendarClock}
          label="Scheduled"
          value={status?.cron.count ?? "—"}
          sub={status ? `${status.cron.enabled} active` : undefined}
        />
        <StatCard
          icon={Radio}
          label="Channels"
          value={
            status ? Object.values(status.channels).filter(Boolean).length : "—"
          }
          sub="connected"
        />
        <StatCard
          icon={Activity}
          label="Active sessions"
          value={status?.active_sessions ?? "—"}
        />
      </div>

      {/* Runtime + channels */}
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <Cpu size={16} className="text-violet-400" /> Runtime
          </div>
          <dl className="flex flex-col gap-2 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="text-zinc-500">Model</dt>
              <dd className="truncate font-mono text-zinc-200">{status?.model ?? "—"}</dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-zinc-500">Fast / Strong</dt>
              <dd className="truncate font-mono text-xs text-zinc-400">
                {status ? `${status.fast_model} · ${status.strong_model}` : "—"}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="flex items-center gap-1.5 text-zinc-500">
                <HardDrive size={13} /> Sandbox
              </dt>
              <dd className="font-mono text-zinc-200">{status?.sandbox ?? "—"}</dd>
            </div>
          </dl>
        </div>

        <div className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-300">
            <Radio size={16} className="text-violet-400" /> Messaging channels
          </div>
          <div className="flex flex-col gap-2">
            {CHANNEL_META.map(({ key, label, icon: Icon }) => {
              const on = status?.channels[key] ?? false;
              return (
                <div key={key} className="flex items-center gap-2 text-sm">
                  <Icon size={15} className="text-zinc-400" />
                  <span className="text-zinc-300">{label}</span>
                  <span
                    className={`ml-auto rounded-full px-2 py-0.5 text-[11px] font-medium ${
                      on ? "bg-emerald-500/15 text-emerald-300" : "bg-zinc-700/40 text-zinc-500"
                    }`}
                  >
                    {on ? "connected" : "off"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
