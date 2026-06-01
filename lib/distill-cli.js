const fs = require("fs");
const http = require("http");
const https = require("https");
const net = require("net");
const os = require("os");
const path = require("path");
const readline = require("readline");
const { spawn, spawnSync } = require("child_process");
const { Writable } = require("stream");

const DEFAULT_REPO_URL = "https://github.com/Aspct3434/agent-ai.git";
const DEFAULT_BRANCH = "master";
const CONFIG_DIR = path.join(os.homedir(), ".distill");
const CONFIG_FILE = path.join(CONFIG_DIR, "config.json");
const RUN_STATE_FILE = path.join(CONFIG_DIR, "run-state.json");

const PROVIDERS = ["kimi", "ollama", "openrouter", "openai", "anthropic", "gemini", "vllm"];
const MESSAGING_CHOICES = ["none", "telegram", "discord", "slack", "email", "all"];
const SANDBOX_CHOICES = ["on", "off"];
const MEMORY_CHOICES = ["lite", "hybrid"];

// Curated model menus per provider. The first entry is the default (and must
// match defaultProviderSettings so non-interactive --yes installs are stable).
// "__custom__" lets the user type any LiteLLM model string instead.
const MODEL_CHOICES = {
  kimi: [
    { label: "Kimi K2.6", value: "moonshot/kimi-k2.6", note: "default" },
    { label: "Kimi K2.5", value: "moonshot/kimi-k2.5" }
  ],
  ollama: [
    { label: "Llama 3.2", value: "ollama/llama3.2", note: "default, small" },
    { label: "Llama 3.3 70B", value: "ollama/llama3.3:70b", note: "needs a strong GPU" },
    { label: "Qwen 2.5 14B", value: "ollama/qwen2.5:14b" }
  ],
  openrouter: [
    { label: "Llama 3.3 70B", value: "openrouter/meta-llama/llama-3.3-70b-instruct", note: "default, open" },
    { label: "Claude Sonnet 4.6", value: "openrouter/anthropic/claude-sonnet-4-6" },
    { label: "Claude Opus 4.8", value: "openrouter/anthropic/claude-opus-4-8", note: "most capable" },
    { label: "GPT-4o", value: "openrouter/openai/gpt-4o" }
  ],
  openai: [
    { label: "GPT-4o", value: "gpt-4o", note: "default" },
    { label: "GPT-4o mini", value: "gpt-4o-mini", note: "cheaper, faster" },
    { label: "GPT-4.1", value: "gpt-4.1" },
    { label: "GPT-4.1 mini", value: "gpt-4.1-mini" },
    { label: "o3", value: "o3", note: "reasoning" },
    { label: "o4-mini", value: "o4-mini", note: "reasoning, cheaper" },
    { label: "o3-mini", value: "o3-mini", note: "reasoning" }
  ],
  anthropic: [
    { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6", note: "default" },
    { label: "Claude Opus 4.8", value: "claude-opus-4-8", note: "most capable" },
    { label: "Claude Haiku 4.5", value: "claude-haiku-4-5", note: "fast, cheap" },
    { label: "Claude Sonnet 4.5", value: "claude-sonnet-4-5" },
    { label: "Claude Opus 4.1", value: "claude-opus-4-1" },
    { label: "Claude Haiku 3.5", value: "claude-3-5-haiku-latest" }
  ],
  gemini: [
    { label: "Gemini 2.5 Flash", value: "gemini/gemini-2.5-flash", note: "default" },
    { label: "Gemini 2.5 Pro", value: "gemini/gemini-2.5-pro", note: "most capable" },
    { label: "Gemini 2.0 Flash", value: "gemini/gemini-2.0-flash" },
    { label: "Gemini 2.0 Flash Lite", value: "gemini/gemini-2.0-flash-lite", note: "fast, cheap" }
  ]
};

function packageVersion() {
  try {
    const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "package.json"), "utf8"));
    return packageJson.version || "unknown";
  } catch {
    return "unknown";
  }
}

const MANAGED_ENV_KEYS = [
  "AGENT_MODEL",
  "FAST_AGENT_MODEL",
  "STRONG_AGENT_MODEL",
  "AGENT_ACTION_MAX_REACT_ITERATIONS",
  "AGENT_MAX_AUTO_CONTINUE_BATCHES",
  "AGENT_MAX_TOKENS",
  "AGENT_FINAL_MAX_TOKENS",
  "AGENT_SANDBOX",
  "AGENT_SANDBOX_HOST_FALLBACK",
  "PUBLIC_BASE_URL",
  "AGENT_USE_HYBRID_MEMORY",
  "MOONSHOT_API_KEY",
  "MOONSHOT_API_BASE",
  "OPENROUTER_API_KEY",
  "OPENAI_API_KEY",
  "OPENAI_API_BASE",
  "ANTHROPIC_API_KEY",
  "GEMINI_API_KEY",
  "OLLAMA_API_BASE",
  "TELEGRAM_BOT_TOKEN",
  "TELEGRAM_ALLOWED_IDS",
  "DISCORD_BOT_TOKEN",
  "DISCORD_ALLOWED_USER_IDS",
  "SLACK_BOT_TOKEN",
  "SLACK_APP_TOKEN",
  "SLACK_ALLOWED_USERS",
  "EMAIL_ADDRESS",
  "EMAIL_PASSWORD",
  "EMAIL_IMAP_HOST",
  "EMAIL_SMTP_HOST",
  "EMAIL_ALLOWED_SENDERS"
];

// ─── ANSI helpers ────────────────────────────────────────────────────────────

function color(code, text) {
  if (!process.stdout.isTTY || process.env.NO_COLOR) return text;
  return `\u001b[${code}m${text}\u001b[0m`;
}

const c = {
  red: (text) => color(31, text),
  green: (text) => color(32, text),
  yellow: (text) => color(33, text),
  blue: (text) => color(34, text),
  magenta: (text) => color(35, text),
  cyan: (text) => color(36, text),
  dim: (text) => color(2, text),
  bold: (text) => color(1, text),
  italic: (text) => color(3, text),
  underline: (text) => color(4, text)
};

function hex(hexStr, text) {
  if (!process.stdout.isTTY || process.env.NO_COLOR) return text;
  const r = parseInt(hexStr.slice(1, 3), 16);
  const g = parseInt(hexStr.slice(3, 5), 16);
  const b = parseInt(hexStr.slice(5, 7), 16);
  return `\u001b[38;2;${r};${g};${b}m${text}\u001b[0m`;
}

function bgHex(hexStr, text) {
  if (!process.stdout.isTTY || process.env.NO_COLOR) return text;
  const r = parseInt(hexStr.slice(1, 3), 16);
  const g = parseInt(hexStr.slice(3, 5), 16);
  const b = parseInt(hexStr.slice(5, 7), 16);
  return `\u001b[48;2;${r};${g};${b}m${text}\u001b[0m`;
}

function stripAnsi(text) {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function padRight(text, width) {
  const visible = stripAnsi(text).length;
  return text + " ".repeat(Math.max(0, width - visible));
}

// ─── Spinner ─────────────────────────────────────────────────────────────────

class Spinner {
  constructor(message) {
    this.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    this.message = message;
    this.index = 0;
    this.timer = null;
  }

  start() {
    if (!process.stdout.isTTY) {
      process.stdout.write(`  ${this.message}\n`);
      return this;
    }
    this.timer = setInterval(() => {
      const frame = hex("#A855F7", this.frames[this.index % this.frames.length]);
      process.stdout.write(`\r  ${frame} ${this.message}`);
      this.index++;
    }, 80);
    return this;
  }

  succeed(text) {
    if (this.timer) clearInterval(this.timer);
    const msg = text || this.message;
    if (process.stdout.isTTY) process.stdout.write(`\r\u001b[2K`);
    process.stdout.write(`  ${hex("#10B981", "✔")} ${msg}\n`);
  }

  fail(text) {
    if (this.timer) clearInterval(this.timer);
    const msg = text || this.message;
    if (process.stdout.isTTY) process.stdout.write(`\r\u001b[2K`);
    process.stdout.write(`  ${hex("#EF4444", "✖")} ${msg}\n`);
  }
}

// ─── UI components ───────────────────────────────────────────────────────────

function box(title, body, options = {}) {
  const accent = options.accent || "#A855F7";
  const bodyLines = body.trim().split(/\r?\n/);
  const titleLine = title ? ` ${title} ` : "";
  const width = Math.max(titleLine.length, ...bodyLines.map((line) => stripAnsi(line).length), 58);
  const top = hex(accent, `╭${"─".repeat(width + 2)}╮`);
  const bottom = hex(accent, `╰${"─".repeat(width + 2)}╯`);
  const side = hex(accent, "│");
  const lines = [top];
  if (titleLine) {
    lines.push(`${side} ${padRight(c.bold(hex(accent, titleLine)), width)} ${side}`);
    lines.push(`${side} ${" ".repeat(width)} ${side}`);
  }
  for (const line of bodyLines) {
    lines.push(`${side} ${padRight(line, width)} ${side}`);
  }
  lines.push(bottom);
  return lines.join("\n");
}

function divider(label) {
  const width = 60;
  if (!label) return hex("#6B7280", "─".repeat(width));
  const labelLen = stripAnsi(label).length;
  const left = 3;
  const right = Math.max(1, width - left - labelLen - 2);
  return hex("#6B7280", "─".repeat(left)) + ` ${hex("#A855F7", label)} ` + hex("#6B7280", "─".repeat(right));
}

function banner() {
  const art = [
    "    ██████╗  ██╗ ███████╗████████╗██╗██╗     ██╗     ",
    "    ██╔══██╗ ██║ ██╔════╝╚══██╔══╝██║██║     ██║     ",
    "    ██║  ██║ ██║ ███████╗   ██║   ██║██║     ██║     ",
    "    ██║  ██║ ██║ ╚════██║   ██║   ██║██║     ██║     ",
    "    ██████╔╝ ██║ ███████║   ██║   ██║███████╗███████╗",
    "    ╚═════╝  ╚═╝ ╚══════╝   ╚═╝   ╚═╝╚══════╝╚══════╝"
  ];

  const colors = ["#8B5CF6", "#7C3AED", "#A855F7", "#9333EA", "#C084FC", "#7C3AED"];

  const lines = [""];
  art.forEach((line, i) => {
    lines.push(c.bold(hex(colors[i], line)));
  });
  lines.push("");
  lines.push(`      ${c.bold(hex("#A855F7", "DISTILL"))}`);
  lines.push(`      ${hex("#6B7280", "v" + packageVersion())}  ${c.dim("·")}  ${hex("#A855F7", "Autonomous Agent Framework")}`);
  lines.push(`      ${hex("#6B7280", "Evidence-gated execution · Skill distillation · Hybrid memory")}`);
  lines.push("");
  return lines.join("\n");
}

function securityWarning() {
  return box(
    "⚠ Security Notice",
    `
${c.bold("Please read before continuing.")}

Distill can execute shell commands, call LLM APIs, start services,
and connect to messaging platforms. A bad prompt, unsafe tool, or
leaked bot token can cause ${c.bold("real damage")}.

${hex("#10B981", "Recommended baseline:")}
  ${hex("#6B7280", "•")} Keep sandbox mode ${c.bold("on")} unless you know why you are disabling it.
  ${hex("#6B7280", "•")} Use allowlists for Telegram / Discord / Slack bots.
  ${hex("#6B7280", "•")} Keep API keys and bot tokens out of public repositories.
  ${hex("#6B7280", "•")} Set ${c.bold("AGENT_API_TOKEN")} when exposing the API beyond localhost.
`,
    { accent: "#F59E0B" }
  );
}

function usage() {
  return `
${c.bold("Distill")} — installer and lifecycle CLI

${hex("#A855F7", "Usage:")}
  distill                         Launch the interactive Terminal UI
  distill install [options]       Interactive guided setup
  distill doctor  [options]       Health check
  distill start   [options]       Start backend + control panel
  distill stop    [options]       Stop all services
  distill restart [options]       Restart all services
  distill update  [options]       Pull latest + restart
  distill logs    [options]       View service logs
  distill tui     [options]       Launch the Terminal UI

${hex("#A855F7", "Install options:")}
  --repo-url URL         Git repository URL
  --branch NAME          Branch to checkout
  --install-dir PATH     Where to clone the project
  --provider NAME        ${c.dim("kimi|ollama|openrouter|openai|anthropic|gemini|vllm")}
  --sandbox on|off       Docker isolation or host-local mode
  --messaging NAME       ${c.dim("none|telegram|discord|slack|email|all")}
  --memory lite|hybrid   ${c.dim("lite (default) or local hybrid memory + embeddings")}
  --mode quickstart|manual
  --dry-run              Print planned commands without executing
  --no-start             Setup files only, don't start services
  --yes                  Accept all defaults non-interactively

${hex("#A855F7", "Lifecycle options:")}
  --install-dir PATH     Override install location
  --sandbox on|off       Override sandbox mode
  --tail NUMBER          Number of log lines to show ${c.dim("(default: 200)")}
  --follow               Stream logs continuously
  --url URL              Gateway WebSocket URL for the TUI
  --theme NAME           TUI theme ${c.dim("(distill|ocean|ember|mono)")}

${hex("#A855F7", "Examples:")}
  ${c.dim("$")} distill
  ${c.dim("$")} npx @aspct3434/distill-agent install
  ${c.dim("$")} npx @aspct3434/distill-agent doctor
  ${c.dim("$")} npx @aspct3434/distill-agent logs --tail 100 --follow
`;
}

function parseArgs(argv) {
  const result = { command: "tui", options: {} };
  const args = [...argv];
  if (args.length > 0 && !args[0].startsWith("-")) {
    result.command = args.shift();
  }
  const valueOptions = new Set([
    "repo-url",
    "branch",
    "install-dir",
    "provider",
    "sandbox",
    "messaging",
    "memory",
    "mode",
    "tail",
    "url",
    "theme"
  ]);
  const booleanOptions = new Set([
    "dry-run",
    "no-start",
    "no-update",
    "yes",
    "follow",
    "help",
    "version"
  ]);
  while (args.length) {
    const arg = args.shift();
    if (!arg.startsWith("--")) throw new Error(`Unknown argument: ${arg}`);
    const [rawName, inlineValue] = arg.slice(2).split("=", 2);
    if (valueOptions.has(rawName)) {
      const value = inlineValue !== undefined ? inlineValue : args.shift();
      if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for --${rawName}`);
      result.options[toCamel(rawName)] = value;
    } else if (booleanOptions.has(rawName)) {
      result.options[toCamel(rawName)] = true;
    } else {
      throw new Error(`Unknown option: --${rawName}`);
    }
  }
  if (result.options.help) result.command = "help";
  if (result.options.version) result.command = "version";
  return result;
}

function toCamel(name) {
  return name.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
}

function normalizeChoice(value, allowed, label) {
  if (value === undefined || value === null || value === "") return "";
  const normalized = String(value).trim().toLowerCase();
  if (!allowed.includes(normalized)) {
    throw new Error(`Invalid ${label}: ${value}. Expected one of: ${allowed.join(", ")}`);
  }
  return normalized;
}

function expandHome(value) {
  if (!value) return value;
  if (value === "~") return os.homedir();
  if (value.startsWith("~/") || value.startsWith("~\\")) return path.join(os.homedir(), value.slice(2));
  return value;
}

function defaultInstallDir() {
  return path.join(os.homedir(), "distill");
}

function envDefault(name, fallback) {
  const value = process.env[name];
  return value && value.trim() ? value : fallback;
}

// ─── Interactive Prompter ────────────────────────────────────────────────────

class Prompter {
  constructor(options = {}) {
    this.input = options.input || process.stdin;
    this.output = options.output || process.stdout;
    this.yes = Boolean(options.yes);
    this.dryRun = Boolean(options.dryRun);
  }

  write(text) {
    this.output.write(text);
  }

  async question(label, defaultValue = "", options = {}) {
    if (this.yes) return defaultValue;
    if (!this.input.isTTY && this.dryRun) return defaultValue;
    if (options.secret) {
      const prompt = defaultValue ? `${hex("#A855F7", "?")} ${c.bold(label)} ${c.dim("[saved value hidden, press Enter to keep]")} ${hex("#A855F7", "❯")} ` : `${hex("#A855F7", "?")} ${c.bold(label)} ${hex("#A855F7", "❯")} `;
      const ans = await this.secretQuestion(prompt, defaultValue);
      this.write(`\u001b[1A\u001b[2K${hex("#10B981", "✔")} ${c.bold(label)} ${c.dim("·")} ${hex("#10B981", "***")}\n`);
      return ans;
    }
    const suffix = defaultValue ? ` ${c.dim("[" + defaultValue + "]")}` : "";
    const prompt = `${hex("#A855F7", "?")} ${c.bold(label)}${suffix} ${hex("#A855F7", "❯")} `;
    return new Promise((resolve) => {
      const rl = readline.createInterface({ input: this.input, output: this.output });
      rl.question(prompt, (answer) => {
        rl.close();
        const finalAns = answer.trim() || defaultValue;
        this.write(`\u001b[1A\u001b[2K${hex("#10B981", "✔")} ${c.bold(label)} ${c.dim("·")} ${hex("#10B981", finalAns)}\n`);
        resolve(finalAns);
      });
    });
  }

  async secretQuestion(prompt, defaultValue = "") {
    if (this.yes) return defaultValue;
    const input = this.input;
    const output = this.output;

    // Non-interactive stdin (piped input, CI, tests): raw-mode masking needs a
    // real terminal, so fall back to a silent readline read.
    if (!input.isTTY || typeof input.setRawMode !== "function") {
      const muted = new Writable({
        write(_chunk, _encoding, callback) {
          callback();
        }
      });
      output.write(prompt);
      return new Promise((resolve) => {
        const rl = readline.createInterface({ input, output: muted, terminal: true });
        rl.question("", (answer) => {
          rl.close();
          output.write("\n");
          resolve(answer.trim() || defaultValue);
        });
      });
    }

    // Interactive terminal: echo a "*" per typed character so the user gets
    // visible feedback without revealing the secret.
    output.write(prompt);
    return new Promise((resolve) => {
      let value = "";
      const wasRaw = Boolean(input.isRaw);
      input.setRawMode(true);
      input.resume();
      input.setEncoding("utf8");

      const finish = (result) => {
        input.setRawMode(wasRaw);
        input.pause();
        input.removeListener("data", onData);
        output.write("\n");
        resolve(result);
      };

      const onData = (chunk) => {
        for (const ch of chunk) {
          const code = ch.charCodeAt(0);
          if (ch === "\r" || ch === "\n") {           // Enter — submit
            finish(value.trim() || defaultValue);
            return;
          }
          if (code === 3) {                            // Ctrl-C — abort
            input.setRawMode(wasRaw);
            input.pause();
            output.write("\n");
            process.exit(130);
          }
          if (code === 4) {                            // Ctrl-D (EOF) — submit
            finish(value.trim() || defaultValue);
            return;
          }
          if (code === 127 || code === 8) {            // Backspace / Delete
            if (value.length > 0) {
              value = value.slice(0, -1);
              output.write("\b \b");
            }
            continue;
          }
          if (code === 27) {                           // ESC — ignore escape
            return;                                    //   sequence (arrows, …)
          }
          if (code >= 0x20) {                          // printable character
            value += ch;
            output.write("*");
          }
        }
      };

      input.on("data", onData);
    });
  }

  async confirm(label, defaultValue = false) {
    if (this.yes) return true;
    const hint = defaultValue ? "Y/n" : "y/N";
    const answer = await this.question(`${label} (${hint})`, "");
    if (!answer) return defaultValue;
    return ["y", "yes", "true", "1", "on"].includes(answer.trim().toLowerCase());
  }

  async select(label, choices, defaultValue) {
    if (this.yes) return defaultValue || choices[0].value;
    this.write(`\n${hex("#A855F7", "?")} ${c.bold(label)}\n`);
    choices.forEach((choice, index) => {
      const isDefault = choice.value === defaultValue;
      const marker = isDefault ? hex("#A855F7", "❯") : " ";
      const num = isDefault ? hex("#A855F7", String(index + 1)) : c.dim(String(index + 1));
      const label = isDefault ? c.bold(choice.label) : choice.label;
      const note = choice.note ? c.dim(` — ${choice.note}`) : "";
      this.write(`  ${marker} ${num}) ${label}${note}\n`);
    });
    const prompt = `  ${hex("#A855F7", "❯")} `;
    let answer = "";
    await new Promise((resolve) => {
      const rl = readline.createInterface({ input: this.input, output: this.output });
      rl.question(prompt, (ans) => {
        rl.close();
        answer = ans.trim();
        resolve();
      });
    });

    let selectedValue = defaultValue || choices[0].value;
    if (answer) {
      const byIndex = Number.parseInt(answer, 10);
      if (Number.isInteger(byIndex) && byIndex >= 1 && byIndex <= choices.length) {
        selectedValue = choices[byIndex - 1].value;
      } else {
        const match = choices.find((choice) => choice.value === answer.trim().toLowerCase());
        if (!match) throw new Error(`Invalid selection: ${answer}`);
        selectedValue = match.value;
      }
    }
    const selectedChoice = choices.find((c) => c.value === selectedValue);
    // Clear the selection UI and replace with a single confirmed line
    for (let i = 0; i < choices.length + 3; i++) {
      this.write("\u001b[1A\u001b[2K");
    }
    this.write(`${hex("#10B981", "✔")} ${c.bold(label)} ${c.dim("·")} ${hex("#10B981", selectedChoice.label)}\n`);
    return selectedValue;
  }
}

// ─── Command helpers ─────────────────────────────────────────────────────────

function commandExists(command) {
  const result = spawnSync(command, ["--version"], { stdio: "ignore", shell: needsShell(command) });
  return !result.error && result.status === 0;
}

function needsShell(command) {
  if (process.platform !== "win32") return false;
  const normalized = command.toLowerCase();
  return normalized === "npm" || normalized === "npx" || normalized === "npm.cmd" || normalized === "npx.cmd";
}

function run(command, args = [], options = {}) {
  const cwd = options.cwd || process.cwd();
  const printable = formatCommand(command, args);
  if (options.dryRun) {
    logStep(`[dry-run] ${printable}`, cwd);
    return Promise.resolve({ code: 0 });
  }
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, ...(options.env || {}) },
      shell: needsShell(command),
      stdio: options.stdio || "inherit",
      windowsHide: true
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve({ code });
      else reject(new Error(`${printable} exited with code ${code}`));
    });
  });
}

function capture(command, args = [], options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || process.cwd(),
    env: { ...process.env, ...(options.env || {}) },
    encoding: "utf8",
    shell: needsShell(command),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });
  return {
    ok: !result.error && result.status === 0,
    status: result.status,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    error: result.error
  };
}

function formatCommand(command, args) {
  return [command, ...args].map((part) => {
    const text = String(part);
    if (/^[A-Za-z0-9_./:=@-]+$/.test(text)) return text;
    return JSON.stringify(text);
  }).join(" ");
}

function logStep(message, cwd) {
  if (cwd && cwd !== process.cwd()) {
    console.log(`  ${hex("#A855F7", "◇")} ${message} ${c.dim(`(${cwd})`)}`);
  } else {
    console.log(`  ${hex("#A855F7", "◇")} ${message}`);
  }
}

function isAgentRepo(dir) {
  return (
    fs.existsSync(path.join(dir, "docker-compose.yml")) &&
    fs.existsSync(path.join(dir, "src", "gateway.py")) &&
    (fs.existsSync(path.join(dir, "scripts", "install.ps1")) || fs.existsSync(path.join(dir, "scripts", "install.sh")))
  );
}

function directoryIsEmpty(dir) {
  if (!fs.existsSync(dir)) return true;
  return fs.readdirSync(dir).length === 0;
}

function readConfig() {
  // Support both new ~/.distill and legacy ~/.agent-ai config locations
  if (fs.existsSync(CONFIG_FILE)) {
    try { return JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8")); } catch { /* fall through */ }
  }
  const legacyConfig = path.join(os.homedir(), ".agent-ai", "config.json");
  if (fs.existsSync(legacyConfig)) {
    try { return JSON.parse(fs.readFileSync(legacyConfig, "utf8")); } catch { /* fall through */ }
  }
  return {};
}

function writeConfig(config, dryRun) {
  if (dryRun) {
    logStep(`[dry-run] write ${CONFIG_FILE}`);
    console.log(JSON.stringify(config, null, 2));
    return;
  }
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(CONFIG_FILE, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}

function readRunState() {
  if (fs.existsSync(RUN_STATE_FILE)) {
    try { return JSON.parse(fs.readFileSync(RUN_STATE_FILE, "utf8")); } catch { /* fall through */ }
  }
  const legacyState = path.join(os.homedir(), ".agent-ai", "run-state.json");
  if (fs.existsSync(legacyState)) {
    try { return JSON.parse(fs.readFileSync(legacyState, "utf8")); } catch { /* fall through */ }
  }
  return {};
}

function writeRunState(state, dryRun) {
  if (dryRun) {
    logStep(`[dry-run] write ${RUN_STATE_FILE}`);
    console.log(JSON.stringify(state, null, 2));
    return;
  }
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(RUN_STATE_FILE, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

async function ensureRepo(config) {
  const installDir = config.installDir;
  if (fs.existsSync(installDir)) {
    if (isAgentRepo(installDir)) {
      logStep(`Found existing Distill checkout at ${installDir}`);
      if (!config.noUpdate) {
        await run("git", ["-C", installDir, "fetch", "origin", config.branch], { dryRun: config.dryRun });
        await run("git", ["-C", installDir, "checkout", config.branch], { dryRun: config.dryRun });
        await run("git", ["-C", installDir, "pull", "--ff-only", "origin", config.branch], { dryRun: config.dryRun });
      }
      return;
    }
    if (!directoryIsEmpty(installDir)) {
      throw new Error(`Install directory exists but is not a Distill checkout: ${installDir}`);
    }
  } else if (config.dryRun) {
    logStep(`[dry-run] create parent directory ${path.dirname(installDir)}`);
  } else {
    fs.mkdirSync(path.dirname(installDir), { recursive: true });
  }
  await run("git", ["clone", "--depth", "1", "--single-branch", "--branch", config.branch, config.repoUrl, installDir], { dryRun: config.dryRun });
}

function requireCommand(command, installHint) {
  if (commandExists(command)) return;
  throw new Error(`${command} is required. ${installHint}`);
}

function checkDockerRunning() {
  const compose = capture("docker", ["compose", "version"]);
  if (!compose.ok) return { ok: false, message: "Docker Compose is not available." };
  const info = capture("docker", ["info"]);
  if (!info.ok) return { ok: false, message: "Docker is installed, but Docker Desktop/daemon is not running." };
  return { ok: true, message: "Docker Compose is available and Docker is running." };
}

function pythonCandidates() {
  const candidates = [];
  if (process.env.AGENT_PYTHON) candidates.push([process.env.AGENT_PYTHON, []]);
  if (process.platform === "win32") {
    candidates.push(["py", ["-3"]], ["python", []]);
  } else {
    candidates.push(["python3.13", []], ["python3.12", []], ["python3.11", []], ["python3", []], ["python", []]);
  }
  return candidates;
}

function pythonMeetsMinimum(command, args = []) {
  return capture(command, [...args, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"]).ok;
}

function pythonHasVenv(command, args = []) {
  let probeParent = null;
  try {
    probeParent = fs.mkdtempSync(path.join(os.tmpdir(), "distill-venv-probe-"));
    return capture(command, [...args, "-m", "venv", path.join(probeParent, "venv")]).ok;
  } finally {
    if (probeParent) fs.rmSync(probeParent, { recursive: true, force: true });
  }
}

function pythonReady() {
  return Boolean(pythonCommand({ requireVenv: true }));
}

function nodeMeetsHostLocalUiMinimum(version = process.version) {
  const match = String(version).trim().match(/^v?(\d+)\.(\d+)\.(\d+)/);
  if (!match) return false;
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (major === 20) return minor >= 19;
  if (major === 22) return minor >= 12;
  return major > 22;
}

function hostLocalNodeRequirementMessage(version = process.version) {
  return `${version} (host-local control panel requires Node.js 20.19+ or 22.12+)`;
}

function withPrivilege(command, args) {
  if (process.platform === "linux" && typeof process.getuid === "function" && process.getuid() !== 0) {
    if (!commandExists("sudo")) {
      throw new Error("sudo is required to auto-install Python. Install Python 3.11+ with venv, then rerun the installer.");
    }
    return ["sudo", [command, ...args]];
  }
  return [command, args];
}

function runSystemCommand(command, args) {
  const [file, commandArgs] = withPrivilege(command, args);
  logStep(formatCommand(file, commandArgs));
  const result = spawnSync(file, commandArgs, {
    stdio: "inherit",
    windowsHide: true
  });
  return !result.error && result.status === 0;
}

function installPythonForLinux() {
  if (process.platform !== "linux") return false;
  logStep("Python 3.11+ with venv is missing; attempting OS package install");

  if (commandExists("apt-get")) {
    if (!runSystemCommand("apt-get", ["update"])) return false;
    const packageSets = [
      ["python3", "python3-venv", "python3-pip"],
      ["python3.13", "python3.13-venv", "python3-pip"],
      ["python3.12", "python3.12-venv", "python3-pip"],
      ["python3.11", "python3.11-venv", "python3-pip"]
    ];
    for (const packages of packageSets) {
      if (runSystemCommand("apt-get", ["install", "-y", ...packages]) && pythonReady()) return true;
    }
    return false;
  }

  if (commandExists("dnf")) {
    const packageSets = [
      ["python3", "python3-pip"],
      ["python3.13", "python3.13-pip"],
      ["python3.12", "python3.12-pip"],
      ["python3.11", "python3.11-pip"]
    ];
    for (const packages of packageSets) {
      if (runSystemCommand("dnf", ["install", "-y", ...packages]) && pythonReady()) return true;
    }
    return false;
  }

  if (commandExists("yum")) {
    const packageSets = [
      ["python3", "python3-pip"],
      ["python3.12", "python3.12-pip"],
      ["python3.11", "python3.11-pip"]
    ];
    for (const packages of packageSets) {
      if (runSystemCommand("yum", ["install", "-y", ...packages]) && pythonReady()) return true;
    }
    return false;
  }

  if (commandExists("pacman")) {
    return runSystemCommand("pacman", ["-Sy", "--noconfirm", "python", "python-pip"]) && pythonReady();
  }

  if (commandExists("apk")) {
    return runSystemCommand("apk", ["add", "--no-cache", "python3", "py3-pip", "py3-virtualenv"]) && pythonReady();
  }

  return false;
}

function ensurePythonForLocalMode() {
  const readySpec = pythonCommand({ requireVenv: true });
  if (readySpec) return readySpec;
  if (installPythonForLinux()) {
    const installedSpec = pythonCommand({ requireVenv: true });
    if (installedSpec) return installedSpec;
  }
  const spec = pythonCommand();
  if (!spec) {
    throw new Error("Python 3.11+ is required for sandbox-off local mode. Install python3.11/python3.12 and the matching venv package, then rerun the installer.");
  }
  throw new Error("Python venv support is required for sandbox-off local mode. Install python3-venv, python3.11-venv, or python3.12-venv, then rerun the installer.");
}

function ensurePrerequisites(config) {
  if (config.dryRun) {
    if (!commandExists("git")) console.log(c.yellow("WARN Git is not available; dry-run will only print planned git commands."));
    return;
  }
  requireCommand("git", "Install Git, reopen your terminal, then rerun the installer.");
  if (config.sandbox === "on" && !config.noStart) {
    const docker = checkDockerRunning();
    if (!docker.ok) throw new Error(`${docker.message} Start Docker, then rerun the installer.`);
  }
  if (config.sandbox === "off" && !config.noStart) {
    ensurePythonForLocalMode();
    requireCommand("npm", "Install Node.js from https://nodejs.org/ and reopen your terminal.");
    if (!nodeMeetsHostLocalUiMinimum()) {
      throw new Error(
        `${hostLocalNodeRequirementMessage()}.\n` +
        "Upgrade Node.js or run with --sandbox on so the control panel builds inside Docker's pinned Node runtime."
      );
    }
  }
}

// ─── Provider & messaging settings ──────────────────────────────────────────

function defaultProviderSettings(provider) {
  switch (provider) {
    case "ollama":
      return {
        agentModel: "ollama/llama3.2",
        fastModel: "ollama/llama3.2",
        strongModel: "ollama/llama3.2",
        ollamaBase: process.env.OLLAMA_API_BASE || ""
      };
    case "openrouter":
      return {
        agentModel: "openrouter/meta-llama/llama-3.3-70b-instruct",
        fastModel: "openrouter/meta-llama/llama-3.1-8b-instruct",
        strongModel: "openrouter/meta-llama/llama-3.3-70b-instruct",
        openrouterKey: ""
      };
    case "openai":
      return {
        agentModel: "gpt-4o",
        fastModel: "gpt-4o",
        strongModel: "gpt-4o",
        openaiKey: ""
      };
    case "anthropic":
      return {
        agentModel: "claude-sonnet-4-6",
        fastModel: "claude-sonnet-4-6",
        strongModel: "claude-sonnet-4-6",
        anthropicKey: ""
      };
    case "gemini":
      return {
        agentModel: "gemini/gemini-2.5-flash",
        fastModel: "gemini/gemini-2.5-flash",
        strongModel: "gemini/gemini-2.5-flash",
        geminiKey: ""
      };
    case "vllm":
      return {
        agentModel: "openai/meta-llama/Llama-3.2-8B-Instruct",
        fastModel: "openai/meta-llama/Llama-3.2-8B-Instruct",
        strongModel: "openai/meta-llama/Llama-3.2-8B-Instruct",
        openaiKey: "dummy",
        openaiBase: process.env.OPENAI_API_BASE || "http://localhost:8001/v1"
      };
    case "kimi":
    default:
      return {
        agentModel: "moonshot/kimi-k2.6",
        fastModel: "moonshot/kimi-k2.6",
        strongModel: "moonshot/kimi-k2.6",
        moonshotKey: "",
        moonshotBase: process.env.MOONSHOT_API_BASE || "https://api.moonshot.ai/v1"
      };
  }
}

async function selectModel(prompter, provider, defaultModel) {
  const choices = MODEL_CHOICES[provider];
  if (!choices) return defaultModel;
  const picked = await prompter.select(
    "Which model should the agent use?",
    [...choices, { label: "Other (enter a model string manually)", value: "__custom__" }],
    defaultModel
  );
  if (picked === "__custom__") {
    return await prompter.question("Model string (LiteLLM format)", defaultModel);
  }
  return picked;
}

async function collectProviderSettings(prompter, provider, mode) {
  const defaults = defaultProviderSettings(provider);
  const settings = { ...defaults };
  if (provider === "vllm") {
    if (mode === "manual") {
      settings.agentModel = await prompter.question("Agent model", defaults.agentModel);
      settings.fastModel = await prompter.question("Fast model", defaults.fastModel);
      settings.strongModel = await prompter.question("Strong model", defaults.strongModel);
    }
  } else {
    const chosen = await selectModel(prompter, provider, defaults.agentModel);
    settings.agentModel = chosen;
    settings.fastModel = chosen;
    settings.strongModel = chosen;
    if (mode === "manual") {
      settings.fastModel = await prompter.question("Fast model", chosen);
      settings.strongModel = await prompter.question("Strong model", chosen);
    }
  }
  if (provider === "kimi") {
    settings.moonshotKey = await prompter.question("Moonshot API key", defaults.moonshotKey, { secret: true });
    if (mode === "manual") settings.moonshotBase = await prompter.question("Moonshot API base", defaults.moonshotBase);
  } else if (provider === "openrouter") {
    settings.openrouterKey = await prompter.question("OpenRouter API key", defaults.openrouterKey, { secret: true });
  } else if (provider === "openai") {
    settings.authMethod = await prompter.select(
      "How should Distill authenticate to OpenAI?",
      [
        { label: "Paste an OpenAI API key", value: "apikey" },
        { label: "Sign in with ChatGPT (Codex OAuth)", value: "oauth", note: "no key; sign in after setup" }
      ],
      "apikey"
    );
    if (settings.authMethod === "apikey") {
      settings.openaiKey = await prompter.question("OpenAI API key", defaults.openaiKey, { secret: true });
    }
  } else if (provider === "anthropic") {
    settings.anthropicKey = await prompter.question("Anthropic API key", defaults.anthropicKey, { secret: true });
  } else if (provider === "gemini") {
    settings.geminiKey = await prompter.question("Gemini API key", defaults.geminiKey, { secret: true });
  } else if (provider === "ollama") {
    settings.ollamaBase = await prompter.question("Ollama API base, blank for default", defaults.ollamaBase);
  } else if (provider === "vllm") {
    settings.openaiKey = await prompter.question("OpenAI-compatible API key", defaults.openaiKey);
    settings.openaiBase = await prompter.question("OpenAI-compatible API base", defaults.openaiBase);
  }
  return settings;
}

async function collectMessagingSettings(prompter, messaging, mode) {
  const settings = {
    telegramToken: "",
    telegramAllowed: process.env.TELEGRAM_ALLOWED_IDS || "",
    discordToken: "",
    discordAllowed: process.env.DISCORD_ALLOWED_USER_IDS || "",
    slackBotToken: "",
    slackAppToken: "",
    slackAllowed: process.env.SLACK_ALLOWED_USERS || "",
    emailAddress: "",
    emailPassword: "",
    emailImapHost: "",
    emailSmtpHost: "",
    emailAllowed: process.env.EMAIL_ALLOWED_SENDERS || ""
  };
  if (messaging === "none") return settings;

  const wantsTelegram = messaging === "telegram" || messaging === "all";
  const wantsDiscord = messaging === "discord" || messaging === "all";
  const wantsSlack = messaging === "slack" || messaging === "all";
  const wantsEmail = messaging === "email" || messaging === "all";

  if (wantsTelegram) {
    settings.telegramToken = await prompter.question("Telegram bot token", settings.telegramToken, { secret: true });
    if (mode === "manual") settings.telegramAllowed = await prompter.question("Telegram allowed chat IDs, blank = all", settings.telegramAllowed);
  }
  if (wantsDiscord) {
    settings.discordToken = await prompter.question("Discord bot token", settings.discordToken, { secret: true });
    if (mode === "manual") settings.discordAllowed = await prompter.question("Discord allowed user IDs, blank = all", settings.discordAllowed);
  }
  if (wantsSlack) {
    settings.slackBotToken = await prompter.question("Slack bot token (xoxb-...)", settings.slackBotToken, { secret: true });
    settings.slackAppToken = await prompter.question("Slack app-level token (xapp-...)", settings.slackAppToken, { secret: true });
    if (mode === "manual") settings.slackAllowed = await prompter.question("Slack allowed user IDs, blank = all", settings.slackAllowed);
  }
  if (wantsEmail) {
    settings.emailAddress = await prompter.question("Email address", settings.emailAddress);
    settings.emailPassword = await prompter.question("Email password / app password", settings.emailPassword, { secret: true });
    settings.emailImapHost = await prompter.question("IMAP host", settings.emailImapHost || "imap.gmail.com");
    settings.emailSmtpHost = await prompter.question("SMTP host", settings.emailSmtpHost || "smtp.gmail.com");
    if (mode === "manual") settings.emailAllowed = await prompter.question("Allowed sender addresses, blank = all", settings.emailAllowed);
  }
  return settings;
}

// ─── Install flow ────────────────────────────────────────────────────────────

async function collectInstallConfig(options) {
  const prompter = new Prompter({ yes: options.yes, dryRun: options.dryRun });
  console.log(banner());
  console.log(divider("ONBOARDING"));
  console.log("");
  console.log(securityWarning());
  console.log("");
  const accepted = await prompter.confirm(hex("#F59E0B", "I understand the risks. Continue?"), false);
  if (!accepted) {
    throw new Error("Installation cancelled.");
  }

  console.log("");
  console.log(divider("SETUP MODE"));
  const mode = normalizeChoice(options.mode, ["quickstart", "manual"], "mode") || await prompter.select(
    "How would you like to configure Distill?",
    [
      { label: "QuickStart", value: "quickstart", note: "guided essentials, recommended defaults" },
      { label: "Manual", value: "manual", note: "full control over every option" }
    ],
    "quickstart"
  );

  let repoUrl = options.repoUrl || envDefault("AGENT_BOOTSTRAP_REPO_URL", DEFAULT_REPO_URL);
  let branch = options.branch || envDefault("AGENT_BOOTSTRAP_BRANCH", DEFAULT_BRANCH);
  let installDir = expandHome(options.installDir || envDefault("AGENT_BOOTSTRAP_INSTALL_DIR", defaultInstallDir()));
  let provider = normalizeChoice(options.provider || process.env.AGENT_INSTALL_PROVIDER, PROVIDERS, "provider") || "kimi";
  let sandbox = normalizeChoice(options.sandbox || process.env.AGENT_INSTALL_SANDBOX, SANDBOX_CHOICES, "sandbox") || "on";
  let messaging = normalizeChoice(options.messaging || process.env.AGENT_INSTALL_MESSAGING, MESSAGING_CHOICES, "messaging") || "none";
  let memory = normalizeChoice(options.memory || process.env.AGENT_INSTALL_MEMORY, MEMORY_CHOICES, "memory") || "lite";

  if (mode === "manual") {
    console.log("");
    console.log(divider("REPOSITORY"));
    repoUrl = await prompter.question("Repository URL", repoUrl);
    branch = await prompter.question("Branch", branch);
    installDir = expandHome(await prompter.question("Install directory", installDir));
  }

  console.log("");
  console.log(divider("LLM PROVIDER"));
  provider = normalizeChoice(await prompter.select(
    "Which LLM provider will you use?",
    [
      { label: "Kimi / Moonshot", value: "kimi", note: "default" },
      { label: "Ollama", value: "ollama", note: "fully local, no API key" },
      { label: "OpenRouter", value: "openrouter", note: "400+ models" },
      { label: "OpenAI", value: "openai" },
      { label: "Anthropic", value: "anthropic" },
      { label: "Google Gemini", value: "gemini" },
      { label: "vLLM / OpenAI-compatible", value: "vllm", note: "self-hosted" }
    ],
    provider
  ), PROVIDERS, "provider");

  console.log("");
  console.log(divider("EXECUTION SANDBOX"));
  sandbox = normalizeChoice(await prompter.select(
    "How should Distill execute commands?",
    [
      { label: "Sandbox ON", value: "on", note: "Docker Compose isolation (recommended)" },
      { label: "Sandbox OFF", value: "off", note: "run directly on this machine" }
    ],
    sandbox
  ), SANDBOX_CHOICES, "sandbox");
  if (sandbox === "off") {
    console.log("");
    console.log(box("⚠ Host-Local Mode", `Sandbox off starts backend and UI processes directly on this PC.\nThe agent will have full access to your local filesystem and shell.\nOnly use this mode if you understand the risk.`, { accent: "#F59E0B" }));
    console.log("");
    const sandboxOffAccepted = await prompter.confirm("Continue with sandbox off?", false);
    if (!sandboxOffAccepted) throw new Error("Installation cancelled before disabling sandbox mode.");
  }

  console.log("");
  console.log(divider("MESSAGING ADAPTERS"));
  messaging = normalizeChoice(await prompter.select(
    "Connect to a messaging platform?",
    [
      { label: "None", value: "none", note: "web UI only" },
      { label: "Telegram", value: "telegram" },
      { label: "Discord", value: "discord" },
      { label: "Slack", value: "slack" },
      { label: "Email", value: "email", note: "IMAP/SMTP" },
      { label: "All of the above", value: "all" }
    ],
    messaging
  ), MESSAGING_CHOICES, "messaging");

  console.log("");
  console.log(divider("MEMORY"));
  memory = normalizeChoice(await prompter.select(
    "Enable local hybrid memory (ChromaDB + Neo4j + embeddings)?",
    [
      { label: "Lite", value: "lite", note: "fast install, no ML deps (recommended)" },
      { label: "Hybrid", value: "hybrid", note: "installs torch/transformers/chromadb (~hundreds of MB)" }
    ],
    memory
  ), MEMORY_CHOICES, "memory");

  console.log("");
  console.log(divider("CREDENTIALS"));
  const providerSettings = await collectProviderSettings(prompter, provider, mode);
  const messagingSettings = await collectMessagingSettings(prompter, messaging, mode);

  const config = {
    repoUrl,
    branch,
    installDir,
    provider,
    sandbox,
    messaging,
    memory,
    mode,
    dryRun: Boolean(options.dryRun),
    noStart: Boolean(options.noStart),
    noUpdate: Boolean(options.noUpdate),
    providerSettings,
    messagingSettings
  };

  console.log("");
  console.log(divider("REVIEW"));
  printInstallSummary(config);
  console.log("");
  const ready = await prompter.confirm(hex("#A855F7", "Start installation with these settings?"), false);
  if (!ready) {
    throw new Error("Installation cancelled before making changes.");
  }
  return config;
}

function printInstallSummary(config) {
  const usesOAuth = config.provider === "openai" && config.providerSettings.authMethod === "oauth";
  const keyStatus = usesOAuth
    ? hex("#A855F7", "Codex OAuth (sign in after setup)")
    : missingCredential(config) ? hex("#F59E0B", "missing") : hex("#10B981", "provided");
  const msgStatus = missingMessagingCredential(config) ? hex("#F59E0B", "missing") : hex("#10B981", "not required or provided");
  console.log(box(
    "Installation Summary",
    [
      `${c.dim("Directory:")}     ${config.installDir}`,
      `${c.dim("Provider:")}      ${config.provider}`,
      `${c.dim("API Key:")}       ${keyStatus}`,
      `${c.dim("Sandbox:")}       ${config.sandbox === "on" ? hex("#10B981", "on (Docker)") : hex("#F59E0B", "off (host)")}`,
      `${c.dim("Messaging:")}     ${config.messaging}`,
      `${c.dim("Memory:")}        ${config.memory === "hybrid" ? hex("#10B981", "hybrid (ML deps)") : "lite"}`,
      `${c.dim("Msg Token:")}     ${msgStatus}`,
      `${c.dim("Start:")}         ${config.noStart ? "no" : hex("#10B981", "yes")}`
    ].join("\n"),
    { accent: "#A855F7" }
  ));
}

function printCodexSignInHelp(config) {
  const venvPy = process.platform === "win32" ? ".run-venv\\Scripts\\python.exe" : ".run-venv/bin/python";
  const lines = [
    "",
    "You chose to sign in with ChatGPT instead of pasting a key.",
    "Sign in once — Distill stores the token and refreshes it for you.",
    "",
    `${hex("#A855F7", "From the control panel:")} Settings → Authentication → Sign in`,
    `${hex("#A855F7", "From the command line:")}`
  ];
  if (config.sandbox === "off") {
    lines.push(`  ${c.dim("$")} cd ${config.installDir}`);
    lines.push(`  ${c.dim("$")} PYTHONPATH=src ${venvPy} -m auth login`);
  } else {
    lines.push(`  ${c.dim("$")} cd ${config.installDir}`);
    lines.push(`  ${c.dim("$")} docker compose exec -e PYTHONPATH=src agent_core python -m auth login`);
    lines.push("");
    lines.push(c.dim("In Docker mode the dashboard sign-in is the simplest route."));
  }
  lines.push("");
  console.log("");
  console.log(box("🔑 Finish Codex OAuth sign-in", lines.join("\n"), { accent: "#A855F7" }));
}

// ─── Env file ────────────────────────────────────────────────────────────────

function envLine(key, value = "") {
  if (value === undefined || value === null || value === "") return `${key}=`;
  const escaped = String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\r?\n/g, "");
  return `${key}="${escaped}"`;
}

function generateEnv(config) {
  const provider = config.providerSettings;
  const messaging = config.messagingSettings;
  const sandboxFallback = config.sandbox === "off" ? "true" : "false";
  return [
    "# Generated by Distill installer",
    `# Sandbox mode: ${config.sandbox}`,
    `# Messaging mode: ${config.messaging}`,
    envLine("AGENT_MODEL", provider.agentModel),
    envLine("FAST_AGENT_MODEL", provider.fastModel),
    envLine("STRONG_AGENT_MODEL", provider.strongModel),
    envLine("AGENT_ACTION_MAX_REACT_ITERATIONS", "18"),
    envLine("AGENT_MAX_AUTO_CONTINUE_BATCHES", "2"),
    envLine("AGENT_MAX_TOKENS", "2048"),
    envLine("AGENT_PLANNING_MAX_TOKENS", "1024"),
    envLine("AGENT_ARTIFACT_MAX_TOKENS", "20000"),
    envLine("AGENT_FINAL_MAX_TOKENS", "1536"),
    envLine("AGENT_SANDBOX", ""),
    envLine("AGENT_SANDBOX_HOST_FALLBACK", sandboxFallback),
    envLine("PUBLIC_BASE_URL", "http://localhost:8000"),
    envLine("AGENT_USE_HYBRID_MEMORY", config.memory === "hybrid" ? "true" : "false"),
    envLine("MOONSHOT_API_KEY", provider.moonshotKey || ""),
    envLine("MOONSHOT_API_BASE", provider.moonshotBase || ""),
    envLine("OPENROUTER_API_KEY", provider.openrouterKey || ""),
    envLine("OPENAI_API_KEY", provider.openaiKey || ""),
    envLine("OPENAI_API_BASE", provider.openaiBase || ""),
    envLine("ANTHROPIC_API_KEY", provider.anthropicKey || ""),
    envLine("GEMINI_API_KEY", provider.geminiKey || ""),
    envLine("OLLAMA_API_BASE", provider.ollamaBase || ""),
    envLine("TELEGRAM_BOT_TOKEN", messaging.telegramToken || ""),
    envLine("TELEGRAM_ALLOWED_IDS", messaging.telegramAllowed || ""),
    envLine("DISCORD_BOT_TOKEN", messaging.discordToken || ""),
    envLine("DISCORD_ALLOWED_USER_IDS", messaging.discordAllowed || ""),
    envLine("SLACK_BOT_TOKEN", messaging.slackBotToken || ""),
    envLine("SLACK_APP_TOKEN", messaging.slackAppToken || ""),
    envLine("SLACK_ALLOWED_USERS", messaging.slackAllowed || ""),
    envLine("EMAIL_ADDRESS", messaging.emailAddress || ""),
    envLine("EMAIL_PASSWORD", messaging.emailPassword || ""),
    envLine("EMAIL_IMAP_HOST", messaging.emailImapHost || ""),
    envLine("EMAIL_SMTP_HOST", messaging.emailSmtpHost || ""),
    envLine("EMAIL_ALLOWED_SENDERS", messaging.emailAllowed || "")
  ].join("\n") + "\n";
}

function writeEnvFile(config) {
  const envPath = path.join(config.installDir, "an-api.env");
  const generated = generateEnv(config);
  if (config.dryRun) {
    logStep(`[dry-run] write ${envPath}`);
    process.stdout.write(generated);
    return;
  }
  fs.mkdirSync(path.dirname(envPath), { recursive: true });
  let preserved = [];
  if (fs.existsSync(envPath)) {
    const backup = `${envPath}.bak.${timestamp()}`;
    fs.copyFileSync(envPath, backup);
    logStep(`Backed up existing env file to ${backup}`);
    const managed = new Set(MANAGED_ENV_KEYS);
    preserved = fs.readFileSync(envPath, "utf8").split(/\r?\n/).filter((line) => {
      const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=/);
      return !match || !managed.has(match[1]);
    });
    while (preserved.length && preserved[preserved.length - 1] === "") preserved.pop();
  }
  const content = preserved.length ? `${preserved.join("\n")}\n\n${generated}` : generated;
  fs.writeFileSync(envPath, content, "utf8");
  logStep(`Wrote ${envPath}`);
}

function missingCredential(config) {
  const provider = config.providerSettings;
  if (config.provider === "kimi" && !provider.moonshotKey) return "MOONSHOT_API_KEY";
  if (config.provider === "openrouter" && !provider.openrouterKey) return "OPENROUTER_API_KEY";
  if (config.provider === "openai" && provider.authMethod !== "oauth" && !provider.openaiKey) return "OPENAI_API_KEY";
  if (config.provider === "anthropic" && !provider.anthropicKey) return "ANTHROPIC_API_KEY";
  if (config.provider === "gemini" && !provider.geminiKey) return "GEMINI_API_KEY";
  return "";
}

function missingMessagingCredential(config) {
  const messaging = config.messagingSettings;
  if ((config.messaging === "telegram" || config.messaging === "all") && !messaging.telegramToken) return "TELEGRAM_BOT_TOKEN";
  if ((config.messaging === "discord" || config.messaging === "all") && !messaging.discordToken) return "DISCORD_BOT_TOKEN";
  if ((config.messaging === "slack" || config.messaging === "all") && !messaging.slackBotToken) return "SLACK_BOT_TOKEN";
  if ((config.messaging === "email" || config.messaging === "all") && !messaging.emailAddress) return "EMAIL_ADDRESS";
  return "";
}

function timestamp() {
  const date = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds())
  ].join("");
}

// ─── Runtime config resolution ───────────────────────────────────────────────

function resolveRuntimeConfig(options = {}) {
  const saved = readConfig();
  const installDir = expandHome(options.installDir || saved.installDir || (isAgentRepo(process.cwd()) ? process.cwd() : defaultInstallDir()));
  const sandbox = normalizeChoice(options.sandbox || saved.sandbox || "on", SANDBOX_CHOICES, "sandbox");
  const memory = normalizeChoice(options.memory || saved.memory || "lite", MEMORY_CHOICES, "memory");
  return {
    installDir,
    sandbox,
    memory,
    repoUrl: options.repoUrl || saved.repoUrl || DEFAULT_REPO_URL,
    branch: options.branch || saved.branch || DEFAULT_BRANCH,
    dryRun: Boolean(options.dryRun),
    noStart: Boolean(options.noStart),
    noUpdate: Boolean(options.noUpdate)
  };
}

function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, "0.0.0.0");
  });
}

function composeServiceIsRunning(config, service) {
  const result = capture("docker", ["compose", "ps", "--status", "running", "-q", service], {
    cwd: config.installDir
  });
  return result.ok && result.stdout.trim().length > 0;
}

async function assertPortsAvailableForCompose(config) {
  if (config.dryRun) return;
  const checks = [
    { port: 8000, service: "agent_core", label: "backend API" },
    { port: 5173, service: "control_panel", label: "control panel" }
  ];
  const blocked = [];
  for (const check of checks) {
    if (composeServiceIsRunning(config, check.service)) continue;
    if (!(await isPortAvailable(check.port))) blocked.push(check);
  }
  if (blocked.length === 0) return;
  const lines = blocked.map((check) => `- Port ${check.port} is already in use, so Distill cannot start the ${check.label}.`);
  throw new Error(`${lines.join("\n")}\n\nIf this is an older Distill stack, run:\n  docker compose -p agentai down\n\nThen clean up:\n  cd ${config.installDir}\n  docker compose down`);
}

async function waitForPortFree(port, timeoutMs = 4000) {
  // Poll until the port is bindable. Absorbs the restart race where a
  // just-stopped process has not released its socket yet.
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await isPortAvailable(port)) return true;
    if (Date.now() >= deadline) return false;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

async function assertPortsAvailableForLocal(config) {
  if (config.dryRun) return;
  const checks = [
    { port: 8000, label: "backend API" },
    { port: 5173, label: "control panel" }
  ];
  const blocked = [];
  for (const check of checks) {
    if (!(await waitForPortFree(check.port))) blocked.push(check);
  }
  if (blocked.length === 0) return;
  const lines = blocked.map(
    (check) => `- Port ${check.port} is already in use, so Distill cannot start the ${check.label}.`
  );
  throw new Error(
    `${lines.join("\n")}\n\n` +
    "Distill (or another process) is already using these ports. Starting again\n" +
    "would spawn a second instance that fails to bind. Restart the running\n" +
    "instance instead:\n" +
    "  npx @aspct3434/distill-agent restart\n" +
    "or stop it first:\n" +
    "  npx @aspct3434/distill-agent stop"
  );
}

async function startServices(config) {
  if (config.sandbox === "on") {
    await assertPortsAvailableForCompose(config);
    const spinner = new Spinner("Starting Distill with Docker Compose...").start();
    try {
      const installMl = config.memory === "hybrid" ? "true" : "false";
      await run("docker", ["compose", "up", "-d", "--build"], { cwd: config.installDir, dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe", env: { INSTALL_ML: installMl, AGENT_USE_HYBRID_MEMORY: installMl } });
      spinner.succeed("Distill is running with Docker Compose");
    } catch (e) {
      spinner.fail("Docker Compose failed to start");
      throw e;
    }
  } else {
    await startLocalServices(config);
    console.log(`  ${hex("#10B981", "✔")} Distill is running in host-local mode`);
  }
  console.log("");
  console.log(box("🚀 Distill is Ready", [
    "",
    `${hex("#A855F7", "Control Panel:")}  ${c.underline("http://localhost:5173")}`,
    `${hex("#A855F7", "API Docs:")}       ${c.underline("http://localhost:8000/docs")}`,
    `${hex("#A855F7", "Health Check:")}   ${c.underline("http://localhost:8000/health")}`,
    `${hex("#A855F7", "Terminal UI:")}    ${c.dim("distill")}`,
    "",
    `${c.dim("Manage with:")}`,
    `  ${c.dim("$")} distill`,
    `  ${c.dim("$")} npx @aspct3434/distill-agent stop`,
    `  ${c.dim("$")} npx @aspct3434/distill-agent logs --follow`,
    `  ${c.dim("$")} npx @aspct3434/distill-agent doctor`,
    ""
  ].join("\n"), { accent: "#10B981" }));
}

function pythonCommand(options = {}) {
  return pythonCandidates().find(([command, args]) => (
    pythonMeetsMinimum(command, args) &&
    (!options.requireVenv || pythonHasVenv(command, args))
  )) || null;
}

function virtualEnvPython(venvDir) {
  return process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

function virtualEnvIsUsable(venvDir) {
  const venvPython = virtualEnvPython(venvDir);
  return fs.existsSync(venvPython) && capture(venvPython, ["-m", "pip", "--version"]).ok;
}

async function ensureProjectVirtualEnv(config, python, pythonPrefix, venvDir) {
  if (config.dryRun) {
    await run(python, [...pythonPrefix, "-m", "venv", ".run-venv"], { cwd: config.installDir, dryRun: true });
    return virtualEnvPython(venvDir);
  }
  if (virtualEnvIsUsable(venvDir)) {
    logStep(`Using existing virtual environment at ${venvDir}`);
    return virtualEnvPython(venvDir);
  }
  if (fs.existsSync(venvDir)) {
    logStep(`Removing incomplete virtual environment at ${venvDir}`);
    fs.rmSync(venvDir, { recursive: true, force: true });
  }
  await run(python, [...pythonPrefix, "-m", "venv", ".run-venv"], { cwd: config.installDir });
  return virtualEnvPython(venvDir);
}

async function startLocalServices(config) {
  // Guard at the single choke point that spawns the services, so no command
  // path (start / restart / update) can launch Vite on an unsupported Node or
  // double-bind a port that an existing instance already holds.
  if (!config.dryRun && !nodeMeetsHostLocalUiMinimum()) {
    throw new Error(
      `${hostLocalNodeRequirementMessage()}.\n` +
      "Upgrade Node.js or run with --sandbox on so the control panel builds inside Docker's pinned Node runtime."
    );
  }
  await assertPortsAvailableForLocal(config);

  const logsDir = path.join(config.installDir, "logs");
  const venvDir = path.join(config.installDir, ".run-venv");
  const pythonSpec = config.dryRun
    ? (pythonCommand() || ["python3", []])
    : ensurePythonForLocalMode();
  if (!pythonSpec) {
    throw new Error("Python 3.11+ is required for sandbox-off local mode.");
  }
  const [python, pythonPrefix] = pythonSpec;
  if (config.dryRun) {
    logStep(`[dry-run] create ${logsDir}`);
  } else {
    fs.mkdirSync(logsDir, { recursive: true });
  }

  const spinnerVenv = new Spinner("Setting up Python virtual environment...").start();
  const venvPython = await ensureProjectVirtualEnv(config, python, pythonPrefix, venvDir);
  spinnerVenv.succeed("Python virtual environment ready");

  const spinnerDeps = new Spinner("Installing Python dependencies...").start();
  // uv resolves and downloads in parallel with a global cache — far faster than
  // pip. Prefer a system uv; otherwise bootstrap it into the venv (tiny wheel);
  // fall back to plain pip if uv can't be installed. The heavy ML stack lives in
  // requirements-ml.txt and is only installed for hybrid memory mode.
  // --index-strategy unsafe-best-match lets uv pick torch's CPU wheel from the
  // extra index declared in requirements-ml.txt.
  const runOpts = { cwd: config.installDir, dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" };
  const reqFiles = ["requirements.txt"];
  if (config.memory === "hybrid") reqFiles.push("requirements-ml.txt");
  const fileArgs = reqFiles.flatMap((f) => ["-r", f]);
  const uvDeps = ["pip", "install", "--index-strategy", "unsafe-best-match", ...fileArgs, "--python", venvPython];
  if (!config.dryRun && capture("uv", ["--version"]).ok) {
    await run("uv", uvDeps, runOpts);
  } else if (!config.dryRun && capture(venvPython, ["-m", "pip", "install", "--upgrade", "pip", "uv"]).ok) {
    await run(venvPython, ["-m", "uv", ...uvDeps], runOpts);
  } else {
    await run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"], runOpts);
    await run(venvPython, ["-m", "pip", "install", ...fileArgs], runOpts);
  }
  spinnerDeps.succeed("Python dependencies installed");

  const spinnerUi = new Spinner("Installing control panel dependencies...").start();
  await run("npm", ["ci", "--no-audit", "--no-fund"], { cwd: path.join(config.installDir, "control-panel"), dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" });
  spinnerUi.succeed("Control panel dependencies installed");

  const backend = spawnDetached(
    venvPython,
    ["-m", "uvicorn", "gateway:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8000"],
    config.installDir,
    path.join(logsDir, "backend-local.stdout.log"),
    path.join(logsDir, "backend-local.stderr.log"),
    config.dryRun
  );
  const frontend = spawnDetached(
    process.platform === "win32" ? "cmd.exe" : "npm",
    process.platform === "win32"
      ? ["/c", "npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
      : ["run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
    path.join(config.installDir, "control-panel"),
    path.join(logsDir, "control-panel-dev.stdout.log"),
    path.join(logsDir, "control-panel-dev.stderr.log"),
    config.dryRun
  );
  writeRunState({
    installDir: config.installDir,
    sandbox: "off",
    backendPid: backend.pid,
    frontendPid: frontend.pid,
    updatedAt: new Date().toISOString()
  }, config.dryRun);
}

function spawnDetached(command, args, cwd, stdoutPath, stderrPath, dryRun) {
  if (dryRun) {
    logStep(`[dry-run] ${formatCommand(command, args)} > ${stdoutPath} 2> ${stderrPath}`, cwd);
    return { pid: 0 };
  }
  if (process.platform === "win32") {
    const psArgs = [
      "-NoProfile", "-NonInteractive", "-Command",
      `$proc = Start-Process -FilePath "${command}" -ArgumentList @(${args.map(a => `"${a.replace(/"/g, '""')}"`).join(", ")}) -WorkingDirectory "${cwd}" -WindowStyle Hidden -PassThru -RedirectStandardOutput "${stdoutPath}" -RedirectStandardError "${stderrPath}"; Write-Output $proc.Id`
    ];
    const result = spawnSync("powershell", psArgs, { windowsHide: true });
    const pidStr = result.stdout ? result.stdout.toString().trim() : "";
    const pid = parseInt(pidStr, 10);
    if (isNaN(pid)) {
      throw new Error(`Failed to start detached process on Windows. Output: ${result.stdout} ${result.stderr}`);
    }
    return { pid };
  }
  const stdout = fs.openSync(stdoutPath, "a");
  const stderr = fs.openSync(stderrPath, "a");
  const child = spawn(command, args, {
    cwd,
    detached: true,
    stdio: ["ignore", stdout, stderr],
    windowsHide: true
  });
  child.unref();
  fs.closeSync(stdout);
  fs.closeSync(stderr);
  return child;
}

async function stopServices(config) {
  if (config.sandbox === "on") {
    const spinner = new Spinner("Stopping Distill services...").start();
    try {
      await run("docker", ["compose", "down"], { cwd: config.installDir, dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" });
      spinner.succeed("Distill services stopped");
    } catch (e) {
      spinner.fail("Failed to stop services");
      throw e;
    }
    return;
  }
  const state = readRunState();
  for (const pid of [state.backendPid, state.frontendPid].filter(Boolean)) {
    await stopPid(pid, config.dryRun);
  }
  if (!config.dryRun && fs.existsSync(RUN_STATE_FILE)) fs.rmSync(RUN_STATE_FILE);
  console.log(`  ${hex("#10B981", "✔")} Distill services stopped`);
}

async function stopPid(pid, dryRun) {
  if (dryRun) {
    logStep(`[dry-run] stop process ${pid}`);
    return;
  }
  if (process.platform === "win32") {
    // /T kills the whole tree (the recorded PID plus children such as Vite).
    await run("taskkill", ["/PID", String(pid), "/T", "/F"], {});
    return;
  }
  // POSIX: startLocalServices spawns each service detached, so the recorded PID
  // is a process-group leader. Signal the whole group (negative PID) -- a bare
  // `kill <npm-pid>` leaves npm's child Vite process orphaned and still holding
  // port 5173. Fall back to the lone PID if it isn't a group leader, and
  // escalate to SIGKILL if the group is still alive after a grace period.
  const signalGroup = (sig) => {
    try {
      process.kill(-pid, sig);
      return true;
    } catch {
      try {
        process.kill(pid, sig);
        return true;
      } catch {
        return false; // already gone
      }
    }
  };
  if (!signalGroup("SIGTERM")) return;
  await new Promise((resolve) => setTimeout(resolve, 600));
  try {
    process.kill(-pid, 0); // throws ESRCH once the group is fully reaped
  } catch {
    return;
  }
  signalGroup("SIGKILL");
}

async function showLogs(config, options) {
  const tail = String(Number.parseInt(options.tail || "200", 10) || 200);
  if (config.sandbox === "on") {
    const args = ["compose", "logs", "--tail", tail];
    if (options.follow) args.push("-f");
    await run("docker", args, { cwd: config.installDir, dryRun: config.dryRun });
    return;
  }
  const logsDir = path.join(config.installDir, "logs");
  for (const fileName of [
    "backend-local.stdout.log",
    "backend-local.stderr.log",
    "control-panel-dev.stdout.log",
    "control-panel-dev.stderr.log"
  ]) {
    const filePath = path.join(logsDir, fileName);
    console.log(hex("#A855F7", `\n══> ${filePath}`));
    if (!fs.existsSync(filePath)) {
      console.log(c.dim("  No log file yet."));
      continue;
    }
    const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
    console.log(lines.slice(-Number(tail)).join("\n"));
  }
}

async function updateRepo(config) {
  if (!isAgentRepo(config.installDir) && !config.dryRun) {
    throw new Error(`Install directory is not a Distill checkout: ${config.installDir}`);
  }
  const spinner = new Spinner("Pulling latest changes...").start();
  try {
    await run("git", ["-C", config.installDir, "fetch", "origin", config.branch], { dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" });
    await run("git", ["-C", config.installDir, "checkout", config.branch], { dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" });
    await run("git", ["-C", config.installDir, "pull", "--ff-only", "origin", config.branch], { dryRun: config.dryRun, stdio: config.dryRun ? "inherit" : "pipe" });
    spinner.succeed("Repository updated");
  } catch (e) {
    spinner.fail("Failed to update repository");
    throw e;
  }
}

async function healthCheck(url, timeoutMs = 1500) {
  return new Promise((resolve) => {
    const client = url.startsWith("https:") ? https : http;
    const request = client.get(url, { timeout: timeoutMs }, (response) => {
      response.resume();
      resolve({ ok: response.statusCode >= 200 && response.statusCode < 500, message: `HTTP ${response.statusCode}` });
    });
    request.on("timeout", () => {
      request.destroy();
      resolve({ ok: false, message: "timeout" });
    });
    request.on("error", (error) => resolve({ ok: false, message: error.message }));
  });
}

// ─── Doctor ──────────────────────────────────────────────────────────────────

async function doctor(options) {
  const config = resolveRuntimeConfig(options);
  console.log("");
  console.log(c.bold(hex("#A855F7", "  Distill Doctor")));
  console.log(divider());
  console.log("");

  const checks = [];
  const nodeOk = config.sandbox !== "off" || nodeMeetsHostLocalUiMinimum();
  checks.push([
    "Node.js",
    nodeOk,
    nodeOk ? process.version : hostLocalNodeRequirementMessage()
  ]);
  checks.push(["npm", commandExists("npm"), commandExists("npm") ? capture("npm", ["--version"]).stdout.trim() : "missing"]);
  checks.push(["Git", commandExists("git"), commandExists("git") ? capture("git", ["--version"]).stdout.trim() : "missing"]);
  checks.push(["Install dir", isAgentRepo(config.installDir), config.installDir]);
  checks.push(["an-api.env", fs.existsSync(path.join(config.installDir, "an-api.env")), path.join(config.installDir, "an-api.env")]);
  const docker = checkDockerRunning();
  checks.push(["Docker", docker.ok, docker.message]);
  const api = await healthCheck("http://localhost:8000/health");
  checks.push(["Backend API", api.ok, api.message]);
  const ui = await healthCheck("http://localhost:5173");
  checks.push(["Control Panel", ui.ok, ui.message]);

  for (const [name, ok, message] of checks) {
    const icon = ok ? hex("#10B981", "✔") : hex("#F59E0B", "⚠");
    const status = ok ? hex("#10B981", "OK") : hex("#F59E0B", "WARN");
    console.log(`  ${icon} ${padRight(c.bold(name), 18)} ${status}  ${c.dim(message)}`);
  }
  console.log("");
  const criticalFailed = checks.some(([name, ok]) => ["Node.js", "npm", "Git", "Install dir"].includes(name) && !ok);
  if (criticalFailed) {
    console.log(hex("#EF4444", "  Some critical checks failed. Fix the issues above and rerun doctor.\n"));
    process.exitCode = 1;
  } else {
    console.log(hex("#10B981", "  All checks passed. Distill is healthy.\n"));
  }
}

// ─── TUI ─────────────────────────────────────────────────────────────────────

function resolveTuiPython(config) {
  for (const name of [".run-venv", "venv"]) {
    const venvDir = path.join(config.installDir, name);
    const candidate = virtualEnvPython(venvDir);
    if (fs.existsSync(candidate)) return [candidate, []];
  }
  const systemPython = pythonCommand();
  if (systemPython && isAgentRepo(config.installDir)) return systemPython;
  throw new Error(
    "Python environment not found. Run 'distill install' first, or run from a Distill checkout with dependencies installed."
  );
}

async function startTui(config, options = {}) {
  if (!config.dryRun && !isAgentRepo(config.installDir)) {
    throw new Error(`Install directory is not a Distill checkout: ${config.installDir}`);
  }
  const [python, pythonPrefix] = config.dryRun
    ? (pythonCommand() || ["python3", []])
    : resolveTuiPython(config);
  const args = [...pythonPrefix, "src/tui.py"];
  if (options.url) args.push("--url", options.url);
  if (options.theme) args.push("--theme", options.theme);

  await run(python, args, { cwd: config.installDir, stdio: "inherit", dryRun: config.dryRun });
}

// ─── Install ─────────────────────────────────────────────────────────────────

async function install(options) {
  const config = await collectInstallConfig(options);
  console.log("");
  console.log(divider("INSTALLING"));
  console.log("");

  // Step 1: Prerequisites
  const spinnerPre = new Spinner("Checking prerequisites...").start();
  try {
    ensurePrerequisites(config);
    spinnerPre.succeed("Prerequisites verified");
  } catch (e) {
    spinnerPre.fail("Prerequisites check failed");
    throw e;
  }

  // Step 2: Clone / update repo
  const spinnerRepo = new Spinner("Setting up repository...").start();
  try {
    await ensureRepo(config);
    spinnerRepo.succeed(`Repository ready at ${config.installDir}`);
  } catch (e) {
    spinnerRepo.fail("Repository setup failed");
    throw e;
  }

  // Step 3: Write env file
  const spinnerEnv = new Spinner("Writing configuration...").start();
  writeEnvFile(config);
  spinnerEnv.succeed("Configuration written to an-api.env");

  // Warnings
  const missingKey = missingCredential(config);
  if (missingKey) {
    console.log(`  ${hex("#F59E0B", "⚠")} ${c.bold(missingKey)} is blank. Add it to ${c.dim(path.join(config.installDir, "an-api.env"))} before using model calls.`);
  }
  const missingBotToken = missingMessagingCredential(config);
  if (missingBotToken) {
    console.log(`  ${hex("#F59E0B", "⚠")} ${c.bold(missingBotToken)} is blank. Messaging integration won't work until you add it.`);
  }

  writeConfig({
    installDir: config.installDir,
    repoUrl: config.repoUrl,
    branch: config.branch,
    sandbox: config.sandbox,
    messaging: config.messaging,
    memory: config.memory,
    provider: config.provider,
    updatedAt: new Date().toISOString()
  }, config.dryRun);

  // Step 4: Start services
  console.log("");
  if (!config.noStart) {
    await startServices(config);
  } else {
    console.log(`  ${hex("#10B981", "✔")} Distill setup files are ready.`);
    console.log(`  ${c.dim("Start later with:")} npx @aspct3434/distill-agent start\n`);
  }

  if (config.provider === "openai" && config.providerSettings.authMethod === "oauth") {
    printCodexSignInHelp(config);
  }
}

// ─── Main entry ──────────────────────────────────────────────────────────────

async function main(argv = process.argv.slice(2)) {
  const { command, options } = parseArgs(argv);
  const implicitTui = argv.length === 0;
  switch (command) {
    case "help":
    case undefined:
      console.log(banner());
      console.log(usage());
      break;
    case "version":
      console.log(packageVersion());
      break;
    case "install":
      await install(options);
      break;
    case "doctor":
      await doctor(options);
      break;
    case "start":
      ensurePrerequisites({ ...resolveRuntimeConfig(options), noStart: false });
      await startServices(resolveRuntimeConfig(options));
      break;
    case "stop":
      await stopServices(resolveRuntimeConfig(options));
      break;
    case "restart": {
      const config = resolveRuntimeConfig(options);
      await stopServices(config);
      ensurePrerequisites({ ...config, noStart: false });
      await startServices(config);
      break;
    }
    case "update": {
      const config = resolveRuntimeConfig(options);
      if (!config.dryRun) requireCommand("git", "Install Git, reopen your terminal, then rerun this command.");
      await updateRepo(config);
      if (!options.noStart) await startServices(config);
      break;
    }
    case "logs":
      await showLogs(resolveRuntimeConfig(options), options);
      break;
    case "tui":
      if (implicitTui && (!process.stdin.isTTY || !process.stdout.isTTY)) {
        console.log(banner());
        console.log(usage());
        break;
      }
      await startTui(resolveRuntimeConfig(options), options);
      break;
    default:
      throw new Error(`Unknown command: ${command}\n${usage()}`);
  }
}

module.exports = {
  main,
  parseArgs,
  generateEnv,
  defaultProviderSettings,
  Prompter,
  DEFAULT_REPO_URL,
  DEFAULT_BRANCH,
  __testing: {
    pythonHasVenv,
    nodeMeetsHostLocalUiMinimum,
    waitForPortFree,
    assertPortsAvailableForLocal
  }
};
