import { useCallback, useEffect, useState } from "react";
import { Cpu, RefreshCw, Save } from "lucide-react";
import { api, type ModelConfig } from "../lib/api";

export function SettingsPanel() {
  const [model, setModel] = useState<ModelConfig>({ model: "", fast_model: "", strong_model: "" });
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    try {
      const m = await api.get<ModelConfig>("/api/config/model");
      setModel(m);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional fetch-on-mount
    void load();
  }, [load]);

  async function saveModel(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSaved(false);
    try {
      const updated = await api.post<ModelConfig & { updated: boolean }>("/api/config/model", model);
      setModel({
        model: updated.model,
        fast_model: updated.fast_model,
        strong_model: updated.strong_model,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const field = (key: keyof ModelConfig, label: string) => (
    <label className="flex flex-col gap-1 text-xs text-zinc-400">
      {label}
      <input
        value={model[key]}
        onChange={(e) => setModel((m) => ({ ...m, [key]: e.target.value }))}
        className="rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 font-mono text-sm text-zinc-100 outline-none focus:border-violet-600"
      />
    </label>
  );

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6">
      <div className="flex items-center gap-3">
        <Cpu className="text-violet-400" size={20} />
        <h2 className="text-lg font-semibold text-zinc-100">Settings</h2>
        <button
          type="button"
          onClick={() => void load()}
          className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-violet-600 hover:text-white"
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-950/30 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}

      <section className="flex flex-col gap-3">
        <h3 className="text-sm font-semibold text-zinc-300">Models</h3>
        <p className="text-xs text-zinc-500">
          Hot-swappable per provider. Use any LiteLLM model string
          (e.g. <span className="font-mono">ollama/llama3.2</span>).
        </p>
        <form onSubmit={saveModel} className="flex flex-col gap-3">
          <div className="grid gap-3 sm:grid-cols-3">
            {field("model", "Main")}
            {field("fast_model", "Fast")}
            {field("strong_model", "Strong")}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              className="flex items-center gap-1.5 rounded-lg bg-violet-700 px-4 py-1.5 text-xs text-white hover:bg-violet-600"
            >
              <Save size={13} /> Save models
            </button>
            {saved && <span className="text-xs text-emerald-400">Saved &amp; hot-swapped.</span>}
          </div>
        </form>
      </section>
    </div>
  );
}
