// Typed client for the agent-ai gateway REST API.

export const API_BASE = "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = (body as { detail?: string }).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PATCH", body: body ? JSON.stringify(body) : undefined }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

// ── Domain types ────────────────────────────────────────────────────────────

export interface Skill {
  name: string;
  file: string;
  description: string;
  tags: string[];
  use_count: number;
  success_rate: number;
  version: number;
  last_used: string | null;
  improved_at: string | null;
}

export interface CronJob {
  job_id: string;
  schedule_type: string;
  schedule_spec: string;
  prompt: string;
  session_id: string;
  label: string;
  deliver_to: string;
  created_at: string;
  next_run: string | null;
  last_run: string | null;
  run_count: number;
  last_result: string;
  enabled: boolean;
}

export interface ModelConfig {
  model: string;
  fast_model: string;
  strong_model: string;
}

export type Health = { status: string };

export interface Tool {
  name: string;
  server: string;
  description: string;
}

export interface Status {
  model: string;
  fast_model: string;
  strong_model: string;
  sandbox: string;
  channels: { telegram: boolean; discord: boolean; slack: boolean };
  skills: { count: number; improved: number };
  cron: { count: number; enabled: number };
  active_sessions: number;
}
