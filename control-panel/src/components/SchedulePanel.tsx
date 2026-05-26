import { useCallback, useEffect, useState } from "react";
import { CalendarClock, Plus, RefreshCw, Trash2 } from "lucide-react";
import { api, type CronJob } from "../lib/api";

function fmtDate(value: string | null): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

const SCHEDULE_TYPES = ["interval", "cron", "once"] as const;

export function SchedulePanel() {
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  const [prompt, setPrompt] = useState("");
  const [scheduleType, setScheduleType] = useState<(typeof SCHEDULE_TYPES)[number]>("interval");
  const [scheduleSpec, setScheduleSpec] = useState("");
  const [label, setLabel] = useState("");
  const [deliverTo, setDeliverTo] = useState("");

  const load = useCallback(async () => {
    try {
      const data = await api.get<CronJob[]>("/api/cron/jobs");
      setJobs(data);
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

  async function createJob(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || !scheduleSpec.trim()) return;
    setError(null);
    try {
      await api.post("/api/cron/jobs", {
        prompt: prompt.trim(),
        schedule_type: scheduleType,
        schedule_spec: scheduleSpec.trim(),
        label: label.trim(),
        deliver_to: deliverTo.trim(),
      });
      setPrompt("");
      setScheduleSpec("");
      setLabel("");
      setDeliverTo("");
      setShowForm(false);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function toggle(job: CronJob) {
    try {
      await api.patch(`/api/cron/jobs/${job.job_id}`, { enabled: !job.enabled });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function remove(jobId: string) {
    try {
      await api.del(`/api/cron/jobs/${jobId}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4">
      <div className="flex items-center gap-3">
        <CalendarClock className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Scheduled tasks</h2>
        <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs text-zinc-400">
          {jobs.length}
        </span>
        <div className="ml-auto flex gap-2">
          <button
            type="button"
            onClick={() => void load()}
            className="flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
          >
            <RefreshCw size={13} /> Refresh
          </button>
          <button
            type="button"
            onClick={() => setShowForm((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg bg-violet-700 px-3 py-1.5 text-xs text-white hover:bg-violet-600"
          >
            <Plus size={13} /> New task
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      {showForm && (
        <form
          onSubmit={createJob}
          className="flex flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-4"
        >
          <textarea
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Task for the agent to run on schedule…"
            className="resize-none rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-violet-600"
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Schedule type
              <select
                value={scheduleType}
                onChange={(e) => setScheduleType(e.target.value as typeof scheduleType)}
                className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-violet-600"
              >
                {SCHEDULE_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Spec
              <input
                value={scheduleSpec}
                onChange={(e) => setScheduleSpec(e.target.value)}
                placeholder={
                  scheduleType === "interval"
                    ? "300  (every 5 min)"
                    : scheduleType === "cron"
                      ? "0 9 * * 1  (Mon 9am)"
                      : "2026-06-01T12:00:00Z"
                }
                className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 font-mono text-sm text-zinc-100 outline-none focus:border-violet-600"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Label (optional)
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-violet-600"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-zinc-400">
              Deliver result to (optional)
              <input
                value={deliverTo}
                onChange={(e) => setDeliverTo(e.target.value)}
                placeholder="tg:12345 / discord:67890 / slack:C123"
                className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 font-mono text-sm text-zinc-100 outline-none focus:border-violet-600"
              />
            </label>
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-lg bg-violet-700 px-4 py-1.5 text-xs text-white hover:bg-violet-600"
            >
              Schedule
            </button>
          </div>
        </form>
      )}

      {loading && <div className="text-sm text-zinc-500">Loading…</div>}
      {!loading && jobs.length === 0 && !showForm && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-8 text-center text-sm text-zinc-500">
          No scheduled tasks. Create one — or just ask the agent in chat to
          "every morning at 8am…" and it'll schedule + deliver it for you.
        </div>
      )}

      <div className="flex flex-col gap-2">
        {jobs.map((job) => (
          <div
            key={job.job_id}
            className="flex flex-col gap-2 rounded-lg border border-zinc-800 bg-zinc-900 p-4"
          >
            <div className="flex items-center gap-2">
              <span className="truncate text-sm font-medium text-zinc-100">
                {job.label || job.prompt.slice(0, 50)}
              </span>
              <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                {job.schedule_type}: {job.schedule_spec}
              </span>
              <button
                type="button"
                onClick={() => void toggle(job)}
                className={`ml-auto rounded-full px-2 py-0.5 text-[11px] font-medium ${
                  job.enabled
                    ? "bg-emerald-500/15 text-emerald-300"
                    : "bg-zinc-700/40 text-zinc-400"
                }`}
              >
                {job.enabled ? "enabled" : "paused"}
              </button>
              <button
                type="button"
                aria-label="Delete task"
                onClick={() => void remove(job.job_id)}
                className="flex size-7 items-center justify-center rounded-md text-zinc-500 hover:bg-zinc-800 hover:text-red-300"
              >
                <Trash2 size={14} />
              </button>
            </div>
            <p className="line-clamp-2 text-xs text-zinc-400">{job.prompt}</p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-zinc-500">
              <span>next: {fmtDate(job.next_run)}</span>
              <span>last: {fmtDate(job.last_run)}</span>
              <span>{job.run_count} runs</span>
              {job.deliver_to && <span>→ {job.deliver_to}</span>}
            </div>
            {job.last_result && (
              <p className="truncate rounded bg-zinc-950/60 px-2 py-1 font-mono text-[11px] text-zinc-400">
                {job.last_result}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
