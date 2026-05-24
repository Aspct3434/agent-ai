#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_URL="https://github.com/Aspct3434/agent-ai.git"
REPO_URL="${AGENT_BOOTSTRAP_REPO_URL:-$DEFAULT_REPO_URL}"
BRANCH="${AGENT_BOOTSTRAP_BRANCH:-master}"
INSTALL_DIR="${AGENT_BOOTSTRAP_INSTALL_DIR:-$HOME/agent-ai}"
PROVIDER="${AGENT_INSTALL_PROVIDER:-}"
SANDBOX="${AGENT_INSTALL_SANDBOX:-}"
MESSAGING="${AGENT_INSTALL_MESSAGING:-}"
DRY_RUN=0
NO_START=0
NO_UPDATE=0

usage() {
  cat <<'EOF'
Agent AI bootstrap installer

Options:
  --repo-url URL
  --branch NAME
  --install-dir PATH
  --provider NAME
  --sandbox on|off
  --messaging none|telegram|discord|both
  --dry-run
  --no-start
  --no-update
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="${2:-}"; shift 2 ;;
    --branch) BRANCH="${2:-}"; shift 2 ;;
    --install-dir) INSTALL_DIR="${2:-}"; shift 2 ;;
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --sandbox) SANDBOX="${2:-}"; shift 2 ;;
    --messaging) MESSAGING="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --no-update) NO_UPDATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

step() { printf '==> %s\n' "$1"; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

run_cmd() {
  local description="$1"; shift
  step "$description"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    return
  fi
  "$@"
}

sudo_if_needed() {
  if [[ "$(id -u)" -eq 0 ]]; then "$@"; elif have_cmd sudo; then sudo "$@"; else return 1; fi
}

ensure_git() {
  if have_cmd git; then step "Git is available"; return; fi
  if have_cmd brew; then run_cmd "Install Git with Homebrew" brew install git; return; fi
  if have_cmd apt-get; then run_cmd "Update apt package lists" sudo_if_needed apt-get update; run_cmd "Install Git with apt" sudo_if_needed apt-get install -y git; return; fi
  if have_cmd dnf; then run_cmd "Install Git with dnf" sudo_if_needed dnf install -y git; return; fi
  if have_cmd yum; then run_cmd "Install Git with yum" sudo_if_needed yum install -y git; return; fi
  echo "Git is required. Install Git, then rerun this bootstrap." >&2
  exit 1
}

ensure_docker() {
  if [[ "$SANDBOX" == "off" ]]; then step "Sandbox off selected; Docker is not required"; return; fi
  if have_cmd docker && docker compose version >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then step "Docker Compose is available and Docker is running"; return; fi
    echo "Docker is installed, but Docker is not running. Start it, then rerun this bootstrap." >&2
    [[ "$DRY_RUN" -eq 1 ]] && return
    exit 1
  fi
  echo "Docker Compose is required for sandbox-on startup. Install Docker, then rerun this bootstrap." >&2
  [[ "$DRY_RUN" -eq 1 ]] && return
  exit 1
}

is_agent_repo() {
  [[ -f "$1/scripts/install.sh" && -f "$1/docker-compose.yml" && -f "$1/src/gateway.py" ]]
}

ensure_repo() {
  if [[ -e "$INSTALL_DIR" ]]; then
    if is_agent_repo "$INSTALL_DIR"; then
      step "Found existing Agent AI checkout at $INSTALL_DIR"
      if [[ "$NO_UPDATE" -ne 1 ]]; then
        run_cmd "Fetch latest repo changes" git -C "$INSTALL_DIR" fetch origin "$BRANCH"
        run_cmd "Checkout selected branch" git -C "$INSTALL_DIR" checkout "$BRANCH"
        run_cmd "Pull selected branch" git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
      fi
      return
    fi
    if [[ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
      echo "Install directory exists but is not an Agent AI checkout: $INSTALL_DIR" >&2
      exit 2
    fi
  else
    run_cmd "Create install parent directory" mkdir -p "$(dirname "$INSTALL_DIR")"
  fi
  run_cmd "Clone Agent AI repository" git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
}

run_installer() {
  local installer="$INSTALL_DIR/scripts/install.sh"
  local args=()
  [[ -n "$PROVIDER" ]] && args+=(--provider "$PROVIDER")
  [[ -n "$SANDBOX" ]] && args+=(--sandbox "$SANDBOX")
  [[ -n "$MESSAGING" ]] && args+=(--messaging "$MESSAGING")
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  [[ "$NO_START" -eq 1 ]] && args+=(--no-start)
  run_cmd "Run Agent AI installer" bash "$installer" "${args[@]}"
}

cat <<EOF
Agent AI bootstrap installer
Install directory: $INSTALL_DIR
Repository:        $REPO_URL
Branch:            $BRANCH

EOF

ensure_git
ensure_docker
ensure_repo
run_installer
