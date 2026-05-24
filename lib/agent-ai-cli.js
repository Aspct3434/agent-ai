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
const CONFIG_DIR = path.join(os.homedir(), ".agent-ai");
const CONFIG_FILE = path.join(CONFIG_DIR, "config.json");
const RUN_STATE_FILE = path.join(CONFIG_DIR, "run-state.json");

const PROVIDERS = ["kimi", "ollama", "openrouter", "openai", "anthropic", "gemini", "vllm"];
const MESSAGING_CHOICES = ["none", "telegram", "discord", "both"];
const SANDBOX_CHOICES = ["on", "off"];

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
  "DISCORD_ALLOWED_USER_IDS"
];

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
  bold: (text) => color(1, text)
};

function stripAnsi(text) {
  return text.replace(/\u001b\[[0-9;]*m/g, "");
}

function padRight(text, width) {
  const visible = stripAnsi(text).length;
  return text + " ".repeat(Math.max(0, width - visible));
}

function box(title, body) {
  const bodyLines = body.trim().split(/\r?\n/);
  const titleLine = title ? `[ ${title} ]` : "";
  const width = Math.max(titleLine.length, ...bodyLines.map((line) => stripAnsi(line).length), 58);
  const top = `+${"-".repeat(width + 2)}+`;
  const lines = [top];
  if (titleLine) {
    lines.push(`| ${padRight(c.red(titleLine), width)} |`);
    lines.push(`| ${" ".repeat(width)} |`);
  }
  for (const line of bodyLines) {
    lines.push(`| ${padRight(line, width)} |`);
  }
  lines.push(top);
  return lines.join("\n");
}

function banner() {
  return [
    "",
    c.bold("    AAAAA   GGGGG  EEEEE  N   N TTTTT      AAAAA  III"),
    c.bold("   A     A G       E      NN  N   T       A     A  I "),
    c.bold("   AAAAAAA G  GGG  EEEE   N N N   T       AAAAAAA  I "),
    c.bold("   A     A G    G  E      N  NN   T       A     A  I "),
    c.bold("   A     A  GGGG   EEEEE  N   N   T       A     A III"),
    "",
    `                  ${c.red("*")} AGENT AI ${c.red("*")}`,
    ""
  ].join("\n");
}

function securityWarning() {
  return box(
    "Security",
    `
Security warning - please read.

Agent AI can execute tools, call models, start services, and connect
to messaging apps when you enable them. A bad prompt, unsafe tool, or
leaked bot token can cause real damage.

Recommended baseline:
- Keep sandbox mode on unless you know why you are disabling it.
- Use allowlists for Telegram or Discord bots.
- Keep API keys and bot tokens out of public repositories.
- Use the strongest model available for any bot with tools enabled.

Run regularly:
npx @aspct3434/agent-ai doctor
npx @aspct3434/agent-ai update
`
  );
}

function usage() {
  return `
Agent AI installer and lifecycle CLI

Usage:
  agent-ai install [options]
  agent-ai doctor [options]
  agent-ai start [options]
  agent-ai stop [options]
  agent-ai restart [options]
  agent-ai update [options]
  agent-ai logs [options]

Install options:
  --repo-url URL
  --branch NAME
  --install-dir PATH
  --provider kimi|ollama|openrouter|openai|anthropic|gemini|vllm
  --sandbox on|off
  --messaging none|telegram|discord|both
  --mode quickstart|manual
  --dry-run
  --no-start
  --no-update
  --yes

Lifecycle options:
  --install-dir PATH
  --sandbox on|off
  --dry-run
  --tail NUMBER
  --follow

Examples:
  npx @aspct3434/agent-ai install
  npx @aspct3434/agent-ai doctor
  npx @aspct3434/agent-ai logs --tail 100
`;
}

function parseArgs(argv) {
  const result = { command: "help", options: {} };
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
    "mode",
    "tail"
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
  return path.join(os.homedir(), "agent-ai");
}

function envDefault(name, fallback) {
  const value = process.env[name];
  return value && value.trim() ? value : fallback;
}

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
      const prompt = defaultValue ? `${label} [saved value hidden, press Enter to keep]: ` : `${label}: `;
      return this.secretQuestion(prompt, defaultValue);
    }
    const suffix = defaultValue ? ` [${defaultValue}]` : "";
    const prompt = `${label}${suffix}: `;
    return new Promise((resolve) => {
      const rl = readline.createInterface({ input: this.input, output: this.output });
      rl.question(prompt, (answer) => {
        rl.close();
        resolve(answer.trim() || defaultValue);
      });
    });
  }

  async secretQuestion(prompt, defaultValue = "") {
    if (this.yes) return defaultValue;
    const muted = new Writable({
      write(_chunk, _encoding, callback) {
        callback();
      }
    });
    this.output.write(prompt);
    return new Promise((resolve) => {
      const rl = readline.createInterface({ input: this.input, output: muted, terminal: true });
      rl.question("", (answer) => {
        rl.close();
        this.output.write("\n");
        resolve(answer.trim() || defaultValue);
      });
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
    this.write(`${label}\n`);
    choices.forEach((choice, index) => {
      const marker = choice.value === defaultValue ? c.green("*") : " ";
      const note = choice.note ? c.dim(` (${choice.note})`) : "";
      this.write(`  ${marker} ${index + 1}) ${choice.label}${note}\n`);
    });
    const answer = await this.question(`Select`, "");
    if (!answer) return defaultValue || choices[0].value;
    const byIndex = Number.parseInt(answer, 10);
    if (Number.isInteger(byIndex) && byIndex >= 1 && byIndex <= choices.length) {
      return choices[byIndex - 1].value;
    }
    const match = choices.find((choice) => choice.value === answer.trim().toLowerCase());
    if (!match) throw new Error(`Invalid selection: ${answer}`);
    return match.value;
  }
}

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
    console.log(`${c.cyan("==>")} ${message} ${c.dim(`(${cwd})`)}`);
  } else {
    console.log(`${c.cyan("==>")} ${message}`);
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
  if (!fs.existsSync(CONFIG_FILE)) return {};
  try {
    return JSON.parse(fs.readFileSync(CONFIG_FILE, "utf8"));
  } catch {
    return {};
  }
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
  if (!fs.existsSync(RUN_STATE_FILE)) return {};
  try {
    return JSON.parse(fs.readFileSync(RUN_STATE_FILE, "utf8"));
  } catch {
    return {};
  }
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
      logStep(`Found existing Agent AI checkout at ${installDir}`);
      if (!config.noUpdate) {
        await run("git", ["-C", installDir, "fetch", "origin", config.branch], { dryRun: config.dryRun });
        await run("git", ["-C", installDir, "checkout", config.branch], { dryRun: config.dryRun });
        await run("git", ["-C", installDir, "pull", "--ff-only", "origin", config.branch], { dryRun: config.dryRun });
      }
      return;
    }
    if (!directoryIsEmpty(installDir)) {
      throw new Error(`Install directory exists but is not an Agent AI checkout: ${installDir}`);
    }
  } else if (config.dryRun) {
    logStep(`[dry-run] create parent directory ${path.dirname(installDir)}`);
  } else {
    fs.mkdirSync(path.dirname(installDir), { recursive: true });
  }
  await run("git", ["clone", "--branch", config.branch, config.repoUrl, installDir], { dryRun: config.dryRun });
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
    const pythonCommands = process.platform === "win32" ? ["py", "python"] : ["python3", "python"];
    if (!pythonCommands.some(commandExists)) {
      throw new Error("Python is required for sandbox-off local mode.");
    }
    requireCommand("npm", "Install Node.js from https://nodejs.org/ and reopen your terminal.");
  }
}

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
        agentModel: "claude-sonnet-4-5",
        fastModel: "claude-sonnet-4-5",
        strongModel: "claude-sonnet-4-5",
        anthropicKey: ""
      };
    case "gemini":
      return {
        agentModel: "gemini/gemini-2.0-flash",
        fastModel: "gemini/gemini-2.0-flash",
        strongModel: "gemini/gemini-2.0-flash",
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

async function collectProviderSettings(prompter, provider, mode) {
  const defaults = defaultProviderSettings(provider);
  const settings = { ...defaults };
  if (mode === "manual") {
    settings.agentModel = await prompter.question("Agent model", defaults.agentModel);
    settings.fastModel = await prompter.question("Fast model", defaults.fastModel);
    settings.strongModel = await prompter.question("Strong model", defaults.strongModel);
  }
  if (provider === "kimi") {
    settings.moonshotKey = await prompter.question("Moonshot API key", defaults.moonshotKey, { secret: true });
    if (mode === "manual") settings.moonshotBase = await prompter.question("Moonshot API base", defaults.moonshotBase);
  } else if (provider === "openrouter") {
    settings.openrouterKey = await prompter.question("OpenRouter API key", defaults.openrouterKey, { secret: true });
  } else if (provider === "openai") {
    settings.openaiKey = await prompter.question("OpenAI API key", defaults.openaiKey, { secret: true });
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
    discordAllowed: process.env.DISCORD_ALLOWED_USER_IDS || ""
  };
  if (messaging === "none") return settings;
  if (messaging === "telegram" || messaging === "both") {
    settings.telegramToken = await prompter.question("Telegram bot token", settings.telegramToken, { secret: true });
    if (mode === "manual") settings.telegramAllowed = await prompter.question("Telegram allowed chat IDs, blank = all", settings.telegramAllowed);
  }
  if (messaging === "discord" || messaging === "both") {
    settings.discordToken = await prompter.question("Discord bot token", settings.discordToken, { secret: true });
    if (mode === "manual") settings.discordAllowed = await prompter.question("Discord allowed user IDs, blank = all", settings.discordAllowed);
  }
  return settings;
}

async function collectInstallConfig(options) {
  const prompter = new Prompter({ yes: options.yes, dryRun: options.dryRun });
  console.log(banner());
  console.log(c.red("Agent AI onboarding"));
  console.log(securityWarning());
  const accepted = await prompter.confirm(c.red("I understand this is powerful and inherently risky. Continue?"), false);
  if (!accepted) {
    throw new Error("Installation cancelled.");
  }

  const mode = normalizeChoice(options.mode, ["quickstart", "manual"], "mode") || await prompter.select(
    c.red("Onboarding mode"),
    [
      { label: "QuickStart", value: "quickstart", note: "guided essentials with recommended defaults" },
      { label: "Manual", value: "manual", note: "choose every option now" }
    ],
    "quickstart"
  );

  let repoUrl = options.repoUrl || envDefault("AGENT_BOOTSTRAP_REPO_URL", DEFAULT_REPO_URL);
  let branch = options.branch || envDefault("AGENT_BOOTSTRAP_BRANCH", DEFAULT_BRANCH);
  let installDir = expandHome(options.installDir || envDefault("AGENT_BOOTSTRAP_INSTALL_DIR", defaultInstallDir()));
  let provider = normalizeChoice(options.provider || process.env.AGENT_INSTALL_PROVIDER, PROVIDERS, "provider") || "kimi";
  let sandbox = normalizeChoice(options.sandbox || process.env.AGENT_INSTALL_SANDBOX, SANDBOX_CHOICES, "sandbox") || "on";
  let messaging = normalizeChoice(options.messaging || process.env.AGENT_INSTALL_MESSAGING, MESSAGING_CHOICES, "messaging") || "none";

  if (mode === "manual") {
    repoUrl = await prompter.question("Repository URL", repoUrl);
    branch = await prompter.question("Branch", branch);
    installDir = expandHome(await prompter.question("Install directory", installDir));
  }

  provider = normalizeChoice(await prompter.select(
    "Model provider",
    [
      { label: "Kimi / Moonshot", value: "kimi" },
      { label: "Ollama", value: "ollama" },
      { label: "OpenRouter", value: "openrouter" },
      { label: "OpenAI", value: "openai" },
      { label: "Anthropic", value: "anthropic" },
      { label: "Gemini", value: "gemini" },
      { label: "vLLM / OpenAI-compatible", value: "vllm" }
    ],
    provider
  ), PROVIDERS, "provider");
  sandbox = normalizeChoice(await prompter.select(
    "Sandbox mode",
    [
      { label: "Sandbox on", value: "on", note: "Docker Compose isolation" },
      { label: "Sandbox off", value: "off", note: "host-local startup" }
    ],
    sandbox
  ), SANDBOX_CHOICES, "sandbox");
  if (sandbox === "off") {
    console.log(box("Host-local warning", "Sandbox off starts backend and UI processes directly on this PC.\nOnly use this mode if you understand the local execution risk."));
    const sandboxOffAccepted = await prompter.confirm("Continue with sandbox off?", false);
    if (!sandboxOffAccepted) throw new Error("Installation cancelled before disabling sandbox mode.");
  }
  messaging = normalizeChoice(await prompter.select(
    "Messaging app",
    [
      { label: "None", value: "none" },
      { label: "Telegram", value: "telegram" },
      { label: "Discord", value: "discord" },
      { label: "Both", value: "both" }
    ],
    messaging
  ), MESSAGING_CHOICES, "messaging");

  const providerSettings = await collectProviderSettings(prompter, provider, mode);
  const messagingSettings = await collectMessagingSettings(prompter, messaging, mode);

  return {
    repoUrl,
    branch,
    installDir,
    provider,
    sandbox,
    messaging,
    mode,
    dryRun: Boolean(options.dryRun),
    noStart: Boolean(options.noStart),
    noUpdate: Boolean(options.noUpdate),
    providerSettings,
    messagingSettings
  };
}

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
    "# Generated by agent-ai npm installer",
    `# Sandbox mode: ${config.sandbox}`,
    `# Messaging mode: ${config.messaging}`,
    envLine("AGENT_MODEL", provider.agentModel),
    envLine("FAST_AGENT_MODEL", provider.fastModel),
    envLine("STRONG_AGENT_MODEL", provider.strongModel),
    envLine("AGENT_ACTION_MAX_REACT_ITERATIONS", "30"),
    envLine("AGENT_MAX_AUTO_CONTINUE_BATCHES", "3"),
    envLine("AGENT_MAX_TOKENS", "32768"),
    envLine("AGENT_FINAL_MAX_TOKENS", "8192"),
    envLine("AGENT_SANDBOX", ""),
    envLine("AGENT_SANDBOX_HOST_FALLBACK", sandboxFallback),
    envLine("PUBLIC_BASE_URL", "http://localhost:8000"),
    envLine("AGENT_USE_HYBRID_MEMORY", "true"),
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
    envLine("DISCORD_ALLOWED_USER_IDS", messaging.discordAllowed || "")
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
  if (config.provider === "openai" && !provider.openaiKey) return "OPENAI_API_KEY";
  if (config.provider === "anthropic" && !provider.anthropicKey) return "ANTHROPIC_API_KEY";
  if (config.provider === "gemini" && !provider.geminiKey) return "GEMINI_API_KEY";
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

function resolveRuntimeConfig(options = {}) {
  const saved = readConfig();
  const installDir = expandHome(options.installDir || saved.installDir || (isAgentRepo(process.cwd()) ? process.cwd() : defaultInstallDir()));
  const sandbox = normalizeChoice(options.sandbox || saved.sandbox || "on", SANDBOX_CHOICES, "sandbox");
  return {
    installDir,
    sandbox,
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
  const lines = blocked.map((check) => `- Port ${check.port} is already in use by another process, so Agent AI cannot start the ${check.label}.`);
  throw new Error(`${lines.join("\n")}\n\nIf this is an older Agent AI stack, run:\n  docker compose -p agentai down\n\nThen clean up this partial install stack:\n  cd ${config.installDir}\n  docker compose down\n\nTo inspect port owners:\n  docker ps --format "table {{.Names}}\\t{{.Ports}}\\t{{.Status}}"`);
}

async function startServices(config) {
  if (config.sandbox === "on") {
    await assertPortsAvailableForCompose(config);
    await run("docker", ["compose", "up", "-d", "--build"], { cwd: config.installDir, dryRun: config.dryRun });
    console.log(c.green(config.dryRun ? "Dry run complete; Docker Compose startup command is shown above." : "Agent AI is starting with Docker Compose."));
  } else {
    await startLocalServices(config);
    console.log(c.green(config.dryRun ? "Dry run complete; host-local startup commands are shown above." : "Agent AI is starting in host-local mode."));
  }
  console.log("Control panel: http://localhost:5173");
  console.log("API health:    http://localhost:8000/health");
}

function pythonCommand() {
  if (process.platform === "win32" && commandExists("py")) return ["py", ["-3"]];
  if (commandExists("python3")) return ["python3", []];
  return ["python", []];
}

async function startLocalServices(config) {
  const logsDir = path.join(config.installDir, "logs");
  const venvDir = path.join(config.installDir, ".run-venv");
  const [python, pythonPrefix] = pythonCommand();
  if (config.dryRun) {
    logStep(`[dry-run] create ${logsDir}`);
  } else {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  await run(python, [...pythonPrefix, "-m", "venv", ".run-venv"], { cwd: config.installDir, dryRun: config.dryRun });
  const venvPython = process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
  await run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"], { cwd: config.installDir, dryRun: config.dryRun });
  await run(venvPython, ["-m", "pip", "install", "-r", "requirements.txt"], { cwd: config.installDir, dryRun: config.dryRun });
  await run("npm", ["ci"], { cwd: path.join(config.installDir, "control-panel"), dryRun: config.dryRun });
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
    await run("docker", ["compose", "down"], { cwd: config.installDir, dryRun: config.dryRun });
    return;
  }
  const state = readRunState();
  for (const pid of [state.backendPid, state.frontendPid].filter(Boolean)) {
    await stopPid(pid, config.dryRun);
  }
  if (!config.dryRun && fs.existsSync(RUN_STATE_FILE)) fs.rmSync(RUN_STATE_FILE);
}

async function stopPid(pid, dryRun) {
  if (dryRun) {
    logStep(`[dry-run] stop process ${pid}`);
    return;
  }
  if (process.platform === "win32") {
    await run("taskkill", ["/PID", String(pid), "/T", "/F"], {});
  } else {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      return;
    }
  }
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
    console.log(c.cyan(`\n==> ${filePath}`));
    if (!fs.existsSync(filePath)) {
      console.log(c.dim("No log file yet."));
      continue;
    }
    const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
    console.log(lines.slice(-Number(tail)).join("\n"));
  }
}

async function updateRepo(config) {
  if (!isAgentRepo(config.installDir) && !config.dryRun) {
    throw new Error(`Install directory is not an Agent AI checkout: ${config.installDir}`);
  }
  await run("git", ["-C", config.installDir, "fetch", "origin", config.branch], { dryRun: config.dryRun });
  await run("git", ["-C", config.installDir, "checkout", config.branch], { dryRun: config.dryRun });
  await run("git", ["-C", config.installDir, "pull", "--ff-only", "origin", config.branch], { dryRun: config.dryRun });
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

async function doctor(options) {
  const config = resolveRuntimeConfig(options);
  console.log(c.bold("\nAgent AI doctor\n"));
  const checks = [];
  checks.push(["Node.js", true, process.version]);
  checks.push(["npm", commandExists("npm"), commandExists("npm") ? capture("npm", ["--version"]).stdout.trim() : "missing"]);
  checks.push(["Git", commandExists("git"), commandExists("git") ? capture("git", ["--version"]).stdout.trim() : "missing"]);
  checks.push(["Install directory", isAgentRepo(config.installDir), config.installDir]);
  checks.push(["an-api.env", fs.existsSync(path.join(config.installDir, "an-api.env")), path.join(config.installDir, "an-api.env")]);
  const docker = checkDockerRunning();
  checks.push(["Docker", docker.ok, docker.message]);
  const api = await healthCheck("http://localhost:8000/health");
  checks.push(["Backend health", api.ok, api.message]);
  const ui = await healthCheck("http://localhost:5173");
  checks.push(["Control panel", ui.ok, ui.message]);
  for (const [name, ok, message] of checks) {
    const status = ok ? c.green("OK") : c.yellow("WARN");
    console.log(`${status} ${padRight(name, 18)} ${message}`);
  }
  const criticalFailed = checks.some(([name, ok]) => ["Node.js", "npm", "Git", "Install directory"].includes(name) && !ok);
  if (criticalFailed) process.exitCode = 1;
}

async function install(options) {
  const config = await collectInstallConfig(options);
  console.log("");
  logStep(`Install directory: ${config.installDir}`);
  logStep(`Repository:        ${config.repoUrl}`);
  logStep(`Branch:            ${config.branch}`);
  logStep(`Sandbox:           ${config.sandbox}`);
  logStep(`Messaging:         ${config.messaging}`);
  ensurePrerequisites(config);
  await ensureRepo(config);
  writeEnvFile(config);
  const missingKey = missingCredential(config);
  if (missingKey) {
    console.log(c.yellow(`WARN ${missingKey} is blank. Add it to ${path.join(config.installDir, "an-api.env")} or rerun manual install before using model calls.`));
  }
  writeConfig({
    installDir: config.installDir,
    repoUrl: config.repoUrl,
    branch: config.branch,
    sandbox: config.sandbox,
    messaging: config.messaging,
    provider: config.provider,
    updatedAt: new Date().toISOString()
  }, config.dryRun);
  if (!config.noStart) await startServices(config);
  else console.log(c.green("Agent AI setup files are ready. Start later with: npx @aspct3434/agent-ai start"));
}

async function main(argv = process.argv.slice(2)) {
  const { command, options } = parseArgs(argv);
  switch (command) {
    case "help":
    case undefined:
      console.log(usage());
      break;
    case "version":
      console.log("0.1.2");
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
    default:
      throw new Error(`Unknown command: ${command}\n${usage()}`);
  }
}

module.exports = {
  main,
  parseArgs,
  generateEnv,
  defaultProviderSettings,
  DEFAULT_REPO_URL,
  DEFAULT_BRANCH
};
