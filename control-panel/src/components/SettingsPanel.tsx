import { useCallback, useEffect, useState } from "react";
import { Cpu, KeyRound, LogIn, LogOut, RefreshCw, Save } from "lucide-react";
import { api, type AuthStatus, type ModelConfig } from "../lib/api";

export function SettingsPanel() {
  const [model, setModel] = useState<ModelConfig>({ model: "", fast_model: "", strong_model: "" });
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [signingIn, setSigningIn] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    try {
      const [m, a] = await Promise.all([
        api.get<ModelConfig>("/api/config/model"),
        api.get<AuthStatus>("/api/auth/status").catch(() => null),
      ]);
      setModel(m);
      setAuth(a);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  async function signIn() {
    setSigningIn(true);
    setError(null);
    try {
      const { authorize_url } = await api.post<{ authorize_url: string }>("/api/auth/login");
      window.open(authorize_url, "_blank", "noopener");
      // Poll status while the browser flow completes.
      for (let i = 0; i < 60; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        const a = await api.get<AuthStatus>("/api/auth/status").catch(() => null);
        if (a?.signed_in) {
          setAuth(a);
          break;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSigningIn(false);
    }
  }

  async function signOut() {
    try {
      await api.post("/api/auth/logout");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

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
        <h3 className="text-sm font-semibold text-zinc-300">Authentication</h3>
        <div className="flex flex-col gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          {auth?.signed_in ? (
            <div className="flex items-center gap-3">
              <span className="flex size-9 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-300">
                <KeyRound size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-sm text-zinc-200">
                  Signed in with OpenAI{auth.email ? ` — ${auth.email}` : ""}
                </div>
                <div className="text-xs text-zinc-500">
                  API key is provided by your account (no manual key needed).
                </div>
              </div>
              <button
                type="button"
                onClick={() => void signOut()}
                className="ml-auto flex items-center gap-1.5 rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:border-red-500/50 hover:text-red-300"
              >
                <LogOut size={13} /> Sign out
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <span className="flex size-9 items-center justify-center rounded-lg bg-zinc-800 text-zinc-400">
                <KeyRound size={16} />
              </span>
              <div className="min-w-0">
                <div className="text-sm text-zinc-200">
                  {auth?.method === "api_key" ? "Using an API key from the environment" : "Not authenticated"}
                </div>
                <div className="text-xs text-zinc-500">
                  Sign in with OpenAI to use your account instead of a pasted key.
                </div>
              </div>
              <button
                type="button"
                onClick={() => void signIn()}
                disabled={signingIn}
                className="ml-auto flex items-center gap-1.5 rounded-lg bg-violet-700 px-3 py-1.5 text-xs text-white hover:bg-violet-600 disabled:opacity-50"
              >
                <LogIn size={13} /> {signingIn ? "Waiting for sign-in…" : "Sign in with OpenAI"}
              </button>
            </div>
          )}
        </div>
      </section>

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
