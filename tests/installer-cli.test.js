const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const bin = path.join(root, "bin", "agent-ai.js");

function run(args, input = "") {
  return spawnSync(process.execPath, [bin, ...args], {
    cwd: root,
    input,
    encoding: "utf8",
    env: {
      ...process.env,
      NO_COLOR: "1"
    }
  });
}

function runWithEnv(args, env, input = "") {
  return spawnSync(process.execPath, [bin, ...args], {
    cwd: root,
    input,
    encoding: "utf8",
    env: {
      ...process.env,
      ...env,
      NO_COLOR: "1"
    }
  });
}

function tempInstallDir(name) {
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "agent-ai-cli-")), name);
}

function assertIncludes(text, expected) {
  assert(
    text.includes(expected),
    `Expected output to include ${JSON.stringify(expected)}.\nActual output:\n${text}`
  );
}

{
  const installDir = tempInstallDir("quickstart");
  const result = run([
    "install",
    "--dry-run",
    "--yes",
    "--no-start",
    "--install-dir",
    installDir,
    "--provider",
    "openai",
    "--sandbox",
    "on",
    "--messaging",
    "none"
  ]);
  assert.strictEqual(result.status, 0, result.stderr);
  assertIncludes(result.stdout, "AGENT AI");
  assertIncludes(result.stdout, "[ Security ]");
  assertIncludes(result.stdout, "[dry-run] git clone --branch master https://github.com/Aspct3434/agent-ai.git");
  assertIncludes(result.stdout, 'AGENT_MODEL="gpt-4o"');
  assertIncludes(result.stdout, 'AGENT_SANDBOX_HOST_FALLBACK="false"');
  assertIncludes(result.stdout, "Start later with: npx @aspct3434/agent-ai start");
}

{
  const installDir = tempInstallDir("manual");
  const result = run([
    "install",
    "--dry-run",
    "--yes",
    "--mode",
    "manual",
    "--no-start",
    "--install-dir",
    installDir,
    "--provider",
    "vllm",
    "--sandbox",
    "off",
    "--messaging",
    "both"
  ]);
  assert.strictEqual(result.status, 0, result.stderr);
  assertIncludes(result.stdout, 'OPENAI_API_BASE="http://localhost:8001/v1"');
  assertIncludes(result.stdout, 'AGENT_SANDBOX_HOST_FALLBACK="true"');
  assertIncludes(result.stdout, "TELEGRAM_BOT_TOKEN=");
  assertIncludes(result.stdout, "DISCORD_BOT_TOKEN=");
}

{
  const installDir = tempInstallDir("no-env-secret-import");
  const result = runWithEnv([
    "install",
    "--dry-run",
    "--yes",
    "--no-start",
    "--install-dir",
    installDir,
    "--provider",
    "openai",
    "--sandbox",
    "on",
    "--messaging",
    "none"
  ], { OPENAI_API_KEY: "sk-should-not-be-imported" });
  assert.strictEqual(result.status, 0, result.stderr);
  assertIncludes(result.stdout, "OPENAI_API_KEY=");
  assert(!result.stdout.includes("sk-should-not-be-imported"));
}

{
  const installDir = tempInstallDir("cancelled");
  const result = run(["install", "--dry-run", "--no-start", "--install-dir", installDir], "n\n");
  assert.notStrictEqual(result.status, 0);
  assertIncludes(result.stderr, "Installation cancelled.");
}

{
  const result = run(["help"]);
  assert.strictEqual(result.status, 0, result.stderr);
  assertIncludes(result.stdout, "agent-ai install");
  assertIncludes(result.stdout, "npx @aspct3434/agent-ai install");
  assertIncludes(result.stdout, "npx @aspct3434/agent-ai doctor");
}

console.log("installer CLI tests passed");
