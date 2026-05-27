// Typed client for the agent-ai gateway REST API.

export const API_BASE = "http://127.0.0.1:8000";

// Optional API token. Only needed when the gateway sets AGENT_API_TOKEN.
// Resolved from localStorage first (lets an operator paste it at runtime),
// falling back to the VITE_AGENT_API_TOKEN build-time env var.
export function getApiToken(): string {
  try {
    const stored = localStorage.getItem("agent_api_token");
    if (stored) return stored;
  } catch {
    /* localStorage unavailable (SSR / privacy mode) */
  }
  return (import.meta.env.VITE_AGENT_API_TOKEN as string | undefined) ?? "";
}

// Append the token to a WebSocket URL when one is configured; browsers can't
// set custom headers on a WS handshake, so the gateway also accepts ?token=.
export function withWsToken(url: string): string {
  const token = getApiToken();
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getApiToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
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
  evolution_status: string;
  version_hash: string;
  last_proof_id: string | null;
  promoted_at: string | null;
  rollback_count: number;
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

export interface AuthStatus {
  method: string; // "oauth" | "api_key" | "none"
  signed_in: boolean;
  email: string;
  account_id: string;
  expires_at: number | null;
  client_id: string;
}

export type Health = { status: string };

export interface Tool {
  name: string;
  server: string;
  description: string;
}

export interface ApprovalRequest {
  id: string;
  command: string;
  session_id: string;
  created_at: number;
}

export interface ApprovalsState {
  mode: string;
  pending: ApprovalRequest[];
}

export interface SessionMatch {
  session_id: string;
  role: string;
  ts: number;
  content: string;
  snippet: string;
}

export interface LogEntry {
  time: number;
  level: string;
  name: string;
  message: string;
}

export interface EvolutionCandidate {
  candidate_id: string;
  kind: "skill" | "prompt_policy" | "toolset_policy";
  name: string;
  status: "staged" | "promoted" | "rejected" | "rolled_back";
  payload: Record<string, unknown>;
  baseline: Record<string, unknown>;
  source_trace_ids: string[];
  proof: {
    passed?: boolean;
    baseline_score?: number;
    candidate_score?: number;
    score_delta?: number;
    checks?: { name: string; passed: boolean; detail?: unknown }[];
  };
  rejection_reason: string;
  rollback: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  promoted_at: string | null;
}

export interface EvolutionStatus {
  db_path: string;
  staging_dir: string;
  verifier_version: string;
  candidates: { staged: number; promoted: number; rejected: number; rolled_back: number };
  active_policies: { prompt_policy: boolean; toolset_policy: boolean };
}

export interface TaskGraphNode {
  id: string;
  title: string;
  kind: string;
  status: "pending" | "ready" | "in_progress" | "done" | "failed" | "blocked";
  depends_on: string[];
  allowed_tools: string[];
  proof_requirements: string[];
  evidence_refs: string[];
  failure_reason: string;
  retry_count: number;
}

export interface TaskGraphVerification {
  passed: boolean;
  missing_nodes: string[];
  invalid_evidence_refs: { node_id: string; evidence_ref: string; reason: string }[];
  blocked_nodes: string[];
  proof_report: Record<string, unknown>[];
  node_count?: number;
}

export interface TaskGraphSnapshot {
  has_graph: boolean;
  source: string | null;
  nodes: TaskGraphNode[];
  active_node: TaskGraphNode | null;
  ready_nodes: TaskGraphNode[];
  blocked_nodes: string[];
  summary?: { total: number; done: number; failed: number; open: number };
  verifier: TaskGraphVerification;
}

export interface PersonaInfo {
  persona_dir: string | null;
  active_persona: string | null;
  available_personas: string[];
  content_length: number;
  loaded: boolean;
}

export interface Status {
  model: string;
  fast_model: string;
  strong_model: string;
  sandbox: string;
  channels: { telegram: boolean; discord: boolean; slack: boolean };
  skills: { count: number; improved: number };
  cron: { count: number; enabled: number };
  evolution: { staged: number; promoted: number; rejected: number; rolled_back: number };
  task_graph: { active: number; blocked: number; open_nodes: number };
  active_sessions: number;
}
