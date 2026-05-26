#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${AGENT_INSTALL_ENV_FILE:-$ROOT_DIR/an-api.env}"
PROVIDER="${AGENT_INSTALL_PROVIDER:-}"
SANDBOX="${AGENT_INSTALL_SANDBOX:-}"
MESSAGING="${AGENT_INSTALL_MESSAGING:-}"
DRY_RUN=0
NO_START=0

usage() {
  cat <<'EOF'
Agent AI installer

Options:
  --provider kimi|ollama|openrouter|openai|anthropic|gemini|vllm
  --sandbox on|off
  --messaging none|telegram|discord|both
  --env-file PATH
  --dry-run
  --no-start
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --sandbox) SANDBOX="${2:-}"; shift 2 ;;
    --messaging) MESSAGING="${2:-}"; shift 2 ;;
    --env-file) ENV_FILE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; NO_START=1; shift ;;
    --no-start) NO_START=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
step() { printf '==> %s\n' "$1" >&2; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

sudo_if_needed() {
  if [[ "$(id -u)" -eq 0 ]]; then "$@"; elif have_cmd sudo; then sudo "$@"; else return 1; fi
}

python_satisfies() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

python_has_venv() {
  local probe_dir status
  probe_dir="$(mktemp -d 2>/dev/null || mktemp -d -t agent-ai-venv-probe)"
  "$1" -m venv "$probe_dir/venv" >/dev/null 2>&1
  status=$?
  rm -rf "$probe_dir"
  return "$status"
}

find_python() {
  local candidate
  if [[ -n "${AGENT_PYTHON:-}" ]] && python_satisfies "$AGENT_PYTHON"; then
    printf '%s' "$AGENT_PYTHON"
    return 0
  fi
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if have_cmd "$candidate" && python_satisfies "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

find_python_with_venv() {
  local candidate
  if [[ -n "${AGENT_PYTHON:-}" ]] && python_satisfies "$AGENT_PYTHON" && python_has_venv "$AGENT_PYTHON"; then
    printf '%s' "$AGENT_PYTHON"
    return 0
  fi
  for candidate in python3.13 python3.12 python3.11 python3 python; do
    if have_cmd "$candidate" && python_satisfies "$candidate" && python_has_venv "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

install_python_with_apt() {
  local packages python_bin
  sudo_if_needed apt-get update || return 1
  for packages in \
    "python3 python3-venv python3-pip" \
    "python3.13 python3.13-venv python3-pip" \
    "python3.12 python3.12-venv python3-pip" \
    "python3.11 python3.11-venv python3-pip"; do
    if sudo_if_needed apt-get install -y $packages; then
      if python_bin="$(find_python_with_venv)"; then return 0; fi
    fi
  done
  return 1
}

install_python_with_dnf() {
  local packages python_bin
  for packages in \
    "python3 python3-pip" \
    "python3.13 python3.13-pip" \
    "python3.12 python3.12-pip" \
    "python3.11 python3.11-pip"; do
    if sudo_if_needed dnf install -y $packages; then
      if python_bin="$(find_python_with_venv)"; then return 0; fi
    fi
  done
  return 1
}

install_python_with_yum() {
  local packages python_bin
  for packages in \
    "python3 python3-pip" \
    "python3.12 python3.12-pip" \
    "python3.11 python3.11-pip"; do
    if sudo_if_needed yum install -y $packages; then
      if python_bin="$(find_python_with_venv)"; then return 0; fi
    fi
  done
  return 1
}

install_python_packages() {
  if have_cmd apt-get; then install_python_with_apt && return 0; fi
  if have_cmd dnf; then install_python_with_dnf && return 0; fi
  if have_cmd yum; then install_python_with_yum && return 0; fi
  if have_cmd pacman; then
    local python_bin
    if sudo_if_needed pacman -Sy --noconfirm python python-pip; then
      if python_bin="$(find_python_with_venv)"; then return 0; fi
    fi
  fi
  if have_cmd apk; then
    local python_bin
    if sudo_if_needed apk add --no-cache python3 py3-pip py3-virtualenv; then
      if python_bin="$(find_python_with_venv)"; then return 0; fi
    fi
  fi
  return 1
}

ensure_python() {
  local python_bin=""
  if python_bin="$(find_python_with_venv)"; then
    printf '%s' "$python_bin"
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'python3'
    return 0
  fi
  step "Python 3.11+ with venv is missing; attempting OS package install" >&2
  if ! install_python_packages >&2; then
    echo "Python 3.11+ with venv is required. Install python3.11-venv or python3.12-venv, then rerun this installer." >&2
    exit 1
  fi
  if ! python_bin="$(find_python)"; then
    echo "Python 3.11+ is required, but the installed python is older or unavailable." >&2
    exit 1
  fi
  if ! python_bin="$(find_python_with_venv)"; then
    echo "Python venv support is required. Install the matching python venv package, then rerun this installer." >&2
    exit 1
  fi
  printf '%s' "$python_bin"
}

project_venv_python() {
  printf '%s' "$ROOT_DIR/.run-venv/bin/python"
}

project_venv_usable() {
  local venv_python
  venv_python="$(project_venv_python)"
  [[ -x "$venv_python" ]] && "$venv_python" -m pip --version >/dev/null 2>&1
}

ensure_project_venv() {
  local python_bin="$1"
  if project_venv_usable; then
    step "Using existing virtual environment at $ROOT_DIR/.run-venv"
    return 0
  fi
  if [[ -d "$ROOT_DIR/.run-venv" ]]; then
    step "Removing incomplete virtual environment at $ROOT_DIR/.run-venv"
    rm -rf "$ROOT_DIR/.run-venv"
  fi
  (cd "$ROOT_DIR" && "$python_bin" -m venv .run-venv)
}

prompt_value() {
  local label="$1" default="${2:-}" answer=""
  if [[ "$DRY_RUN" -eq 1 ]]; then printf '%s' "$default"; return; fi
  if [[ -n "$default" ]]; then
    read -r -p "$label [$default]: " answer
    printf '%s' "${answer:-$default}"
  else
    read -r -p "$label: " answer
    printf '%s' "$answer"
  fi
}

prompt_secret() {
  local env_name="$1" label="$2" current="${!env_name:-}" answer=""
  if [[ -n "$current" ]]; then printf '%s' "$current"; return; fi
  if [[ "$DRY_RUN" -eq 1 ]]; then printf ''; return; fi
  read -r -s -p "$label: " answer
  printf '\n' >&2
  printf '%s' "$answer"
}

choose_provider() {
  local value; value="$(lower "$PROVIDER")"
  case "$value" in kimi|moonshot|ollama|openrouter|openai|anthropic|gemini|vllm) printf '%s' "$value"; return ;; esac
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'kimi'; return; fi
  cat <<'EOF'
Choose model provider:
  1) Kimi / Moonshot
  2) Ollama
  3) OpenRouter
  4) OpenAI
  5) Anthropic
  6) Gemini
  7) vLLM / OpenAI-compatible
EOF
  read -r -p "Provider [1]: " value
  case "${value:-1}" in
    1) printf 'kimi' ;; 2) printf 'ollama' ;; 3) printf 'openrouter' ;;
    4) printf 'openai' ;; 5) printf 'anthropic' ;; 6) printf 'gemini' ;;
    7) printf 'vllm' ;; *) echo "Invalid provider: $value" >&2; exit 2 ;;
  esac
}

choose_sandbox() {
  local value; value="$(lower "$SANDBOX")"
  case "$value" in on|off) printf '%s' "$value"; return ;; esac
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'on'; return; fi
  read -r -p "Sandbox on? Docker Compose container isolation [Y/n]: " value
  case "$(lower "${value:-y}")" in y|yes|on|1) printf 'on' ;; n|no|off|2) printf 'off' ;; *) echo "Invalid sandbox choice" >&2; exit 2 ;; esac
}

choose_messaging() {
  local value; value="$(lower "$MESSAGING")"
  case "$value" in none|telegram|discord|both) printf '%s' "$value"; return ;; esac
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'none'; return; fi
  cat <<'EOF'
Choose messaging app:
  1) None
  2) Telegram
  3) Discord
  4) Both
EOF
  read -r -p "Messaging [1]: " value
  case "${value:-1}" in
    1|none) printf 'none' ;; 2|telegram) printf 'telegram' ;;
    3|discord) printf 'discord' ;; 4|both) printf 'both' ;;
    *) echo "Invalid messaging choice" >&2; exit 2 ;;
  esac
}

env_line() {
  local key="$1" value="${2:-}"
  value="${value//$'\r'/}"
  value="${value//$'\n'/}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  if [[ -z "$value" ]]; then printf '%s=\n' "$key"; else printf '%s="%s"\n' "$key" "$value"; fi
}

provider="$(choose_provider)"
sandbox="$(choose_sandbox)"
messaging="$(choose_messaging)"

agent_model=""; fast_model=""; strong_model=""
moonshot_key=""; moonshot_base=""; openrouter_key=""; openai_key=""; openai_base=""
anthropic_key=""; gemini_key=""; ollama_base=""

case "$provider" in
  kimi|moonshot)
    agent_model="$(prompt_value 'Agent model' 'moonshot/kimi-k2.6')"
    fast_model="$(prompt_value 'Fast model' "$agent_model")"
    strong_model="$(prompt_value 'Strong model' "$agent_model")"
    moonshot_key="$(prompt_secret MOONSHOT_API_KEY 'Moonshot API key')"
    moonshot_base="$(prompt_value 'Moonshot API base' 'https://api.moonshot.ai/v1')"
    ;;
  ollama)
    agent_model="$(prompt_value 'Ollama model' 'ollama/llama3.2')"; fast_model="$agent_model"; strong_model="$agent_model"
    ollama_base="$(prompt_value 'Ollama API base, blank for default' '')"
    ;;
  openrouter)
    agent_model="$(prompt_value 'OpenRouter model' 'openrouter/meta-llama/llama-3.3-70b-instruct')"
    fast_model="$(prompt_value 'Fast model' 'openrouter/meta-llama/llama-3.1-8b-instruct')"
    strong_model="$(prompt_value 'Strong model' "$agent_model")"
    openrouter_key="$(prompt_secret OPENROUTER_API_KEY 'OpenRouter API key')"
    ;;
  openai)
    agent_model="$(prompt_value 'OpenAI model' 'gpt-4o')"; fast_model="$agent_model"; strong_model="$agent_model"
    openai_key="$(prompt_secret OPENAI_API_KEY 'OpenAI API key')"
    ;;
  anthropic)
    agent_model="$(prompt_value 'Anthropic model' 'claude-sonnet-4-5')"; fast_model="$agent_model"; strong_model="$agent_model"
    anthropic_key="$(prompt_secret ANTHROPIC_API_KEY 'Anthropic API key')"
    ;;
  gemini)
    agent_model="$(prompt_value 'Gemini model' 'gemini/gemini-2.0-flash')"; fast_model="$agent_model"; strong_model="$agent_model"
    gemini_key="$(prompt_secret GEMINI_API_KEY 'Gemini API key')"
    ;;
  vllm)
    agent_model="$(prompt_value 'vLLM model' 'openai/meta-llama/Llama-3.2-8B-Instruct')"; fast_model="$agent_model"; strong_model="$agent_model"
    openai_key="$(prompt_value 'OpenAI-compatible API key' 'dummy')"
    openai_base="$(prompt_value 'OpenAI-compatible API base' 'http://localhost:8001/v1')"
    ;;
esac

telegram_token=""; telegram_allowed=""; discord_token=""; discord_allowed=""
case "$messaging" in telegram|both) telegram_token="$(prompt_secret TELEGRAM_BOT_TOKEN 'Telegram bot token')"; telegram_allowed="$(prompt_value 'Telegram allowed chat IDs, blank = all' '')" ;; esac
case "$messaging" in discord|both) discord_token="$(prompt_secret DISCORD_BOT_TOKEN 'Discord bot token')"; discord_allowed="$(prompt_value 'Discord allowed user IDs, blank = all' '')" ;; esac

agent_sandbox=""; sandbox_fallback="false"
if [[ "$sandbox" == "off" ]]; then sandbox_fallback="true"; fi

generated_env() {
  echo "# Generated by scripts/install.sh"
  echo "# Sandbox mode: $sandbox"
  echo "# Messaging mode: $messaging"
  env_line AGENT_MODEL "$agent_model"
  env_line FAST_AGENT_MODEL "$fast_model"
  env_line STRONG_AGENT_MODEL "$strong_model"
  env_line AGENT_ACTION_MAX_REACT_ITERATIONS "30"
  env_line AGENT_MAX_AUTO_CONTINUE_BATCHES "3"
  env_line AGENT_MAX_TOKENS "32768"
  env_line AGENT_FINAL_MAX_TOKENS "8192"
  env_line AGENT_SANDBOX "$agent_sandbox"
  env_line AGENT_SANDBOX_HOST_FALLBACK "$sandbox_fallback"
  env_line PUBLIC_BASE_URL "http://localhost:8000"
  env_line AGENT_USE_HYBRID_MEMORY "true"
  env_line MOONSHOT_API_KEY "$moonshot_key"
  env_line MOONSHOT_API_BASE "$moonshot_base"
  env_line OPENROUTER_API_KEY "$openrouter_key"
  env_line OPENAI_API_KEY "$openai_key"
  env_line OPENAI_API_BASE "$openai_base"
  env_line ANTHROPIC_API_KEY "$anthropic_key"
  env_line GEMINI_API_KEY "$gemini_key"
  env_line OLLAMA_API_BASE "$ollama_base"
  env_line TELEGRAM_BOT_TOKEN "$telegram_token"
  env_line TELEGRAM_ALLOWED_IDS "$telegram_allowed"
  env_line DISCORD_BOT_TOKEN "$discord_token"
  env_line DISCORD_ALLOWED_USER_IDS "$discord_allowed"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "# Dry run: generated env for $ENV_FILE"
  generated_env
  exit 0
fi

managed='AGENT_MODEL|FAST_AGENT_MODEL|STRONG_AGENT_MODEL|AGENT_ACTION_MAX_REACT_ITERATIONS|AGENT_MAX_AUTO_CONTINUE_BATCHES|AGENT_MAX_TOKENS|AGENT_FINAL_MAX_TOKENS|AGENT_SANDBOX|AGENT_SANDBOX_HOST_FALLBACK|PUBLIC_BASE_URL|AGENT_USE_HYBRID_MEMORY|MOONSHOT_API_KEY|MOONSHOT_API_BASE|OPENROUTER_API_KEY|OPENAI_API_KEY|OPENAI_API_BASE|ANTHROPIC_API_KEY|GEMINI_API_KEY|OLLAMA_API_BASE|TELEGRAM_BOT_TOKEN|TELEGRAM_ALLOWED_IDS|DISCORD_BOT_TOKEN|DISCORD_ALLOWED_USER_IDS'
tmp="$(mktemp)"
mkdir -p "$(dirname "$ENV_FILE")"
if [[ -f "$ENV_FILE" ]]; then
  backup="$ENV_FILE.bak.$(date +%Y%m%d%H%M%S)"
  cp "$ENV_FILE" "$backup"
  grep -Ev "^(${managed})=" "$ENV_FILE" > "$tmp" || true
  echo "Backed up existing env file to $backup"
fi
{ [[ -s "$tmp" ]] && cat "$tmp" && echo; generated_env; } > "$ENV_FILE"
rm -f "$tmp"
echo "Wrote $ENV_FILE"

if [[ "$NO_START" -eq 0 ]]; then
  if [[ "$sandbox" == "on" ]]; then
    (cd "$ROOT_DIR" && docker compose up -d --build)
  else
    python_bin="$(ensure_python)"
    mkdir -p "$ROOT_DIR/logs"
    ensure_project_venv "$python_bin"
    (cd "$ROOT_DIR" && .run-venv/bin/python -m pip install --upgrade pip && .run-venv/bin/python -m pip install -r requirements.txt)
    (cd "$ROOT_DIR" && . .run-venv/bin/activate && set -a && . "$ENV_FILE" && set +a && nohup python -m uvicorn gateway:app --app-dir src --host 127.0.0.1 --port 8000 > logs/backend-local.stdout.log 2> logs/backend-local.stderr.log &)
    (cd "$ROOT_DIR/control-panel" && npm ci && nohup npm run dev -- --host 127.0.0.1 --port 5173 > ../logs/control-panel-dev.stdout.log 2> ../logs/control-panel-dev.stderr.log &)
  fi
fi

echo "Agent AI setup complete."
echo "Control panel: http://localhost:5173"
echo "API health:    http://localhost:8000/health"
