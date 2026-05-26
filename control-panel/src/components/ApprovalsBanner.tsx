import { useEffect, useState } from "react";
import { Check, ShieldAlert, X } from "lucide-react";
import { api, type ApprovalRequest, type ApprovalsState } from "../lib/api";

/** Polls for pending command approvals and lets the user approve/deny them.
 * Renders nothing when there's nothing to approve. */
export function ApprovalsBanner() {
  const [items, setItems] = useState<ApprovalRequest[]>([]);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const s = await api.get<ApprovalsState>("/api/approvals");
        if (alive) setItems(s.pending);
      } catch {
        /* gateway down — ignore */
      }
    };
    void poll();
    const id = setInterval(() => void poll(), 2000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  async function decide(id: string, approved: boolean) {
    setItems((x) => x.filter((i) => i.id !== id));
    try {
      await api.post(`/api/approvals/${id}`, { approved });
    } catch {
      /* already resolved/expired — ignore */
    }
  }

  if (items.length === 0) return null;

  return (
    <div className="flex flex-col gap-2 border-b border-amber-500/30 bg-amber-500/10 px-4 py-3">
      {items.map((req) => (
        <div key={req.id} className="flex items-center gap-3">
          <ShieldAlert size={16} className="shrink-0 text-amber-300" />
          <div className="min-w-0 flex-1">
            <div className="text-xs font-medium text-amber-200">
              The agent wants to run a command — approve?
            </div>
            <code className="block truncate font-mono text-xs text-zinc-200">{req.command}</code>
          </div>
          <button
            type="button"
            onClick={() => void decide(req.id, true)}
            className="flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs text-white hover:bg-emerald-500"
          >
            <Check size={13} /> Approve
          </button>
          <button
            type="button"
            onClick={() => void decide(req.id, false)}
            className="flex items-center gap-1 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-red-500/50 hover:text-red-300"
          >
            <X size={13} /> Deny
          </button>
        </div>
      ))}
    </div>
  );
}
