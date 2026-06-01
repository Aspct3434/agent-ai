param(
    [string]$Provider = $env:AGENT_INSTALL_PROVIDER,
    [string]$Sandbox = $env:AGENT_INSTALL_SANDBOX,
    [string]$Messaging = $env:AGENT_INSTALL_MESSAGING,
    [string]$Memory = $env:AGENT_INSTALL_MEMORY,
    [string]$EnvFile = $env:AGENT_INSTALL_ENV_FILE,
    [switch]$DryRun,
    [switch]$NoStart
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $RootDir "an-api.env"
}

function Normalize([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return "" }
    return $Value.Trim().ToLowerInvariant()
}

function Prompt-Value([string]$Prompt, [string]$Default = "") {
    if ($DryRun) { return $Default }
    if ([string]::IsNullOrEmpty($Default)) { return (Read-Host $Prompt) }
    $answer = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrEmpty($answer)) { return $Default }
    return $answer
}

function Prompt-Secret([string]$EnvName, [string]$Prompt) {
    $current = [Environment]::GetEnvironmentVariable($EnvName, "Process")
    if (-not [string]::IsNullOrEmpty($current)) { return $current }
    if ($DryRun) { return "" }
    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

function Choose-Provider {
    $value = Normalize $Provider
    if (@("kimi", "moonshot", "ollama", "openrouter", "openai", "anthropic", "gemini", "vllm") -contains $value) { return $value }
    if ($DryRun) { return "kimi" }
    Write-Host "Choose model provider:"
    Write-Host "  1) Kimi / Moonshot"
    Write-Host "  2) Ollama"
    Write-Host "  3) OpenRouter"
    Write-Host "  4) OpenAI"
    Write-Host "  5) Anthropic"
    Write-Host "  6) Gemini"
    Write-Host "  7) vLLM / OpenAI-compatible"
    $answer = Read-Host "Provider [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    switch ($answer.ToLowerInvariant()) {
        "1" { return "kimi" }
        "2" { return "ollama" }
        "3" { return "openrouter" }
        "4" { return "openai" }
        "5" { return "anthropic" }
        "6" { return "gemini" }
        "7" { return "vllm" }
        default { throw "Invalid provider choice: $answer" }
    }
}

function Choose-Sandbox {
    $value = Normalize $Sandbox
    if (@("on", "off") -contains $value) { return $value }
    if ($DryRun) { return "on" }
    $answer = Read-Host "Sandbox on? Docker Compose container isolation [Y/n]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "y" }
    switch ($answer.ToLowerInvariant()) {
        "y" { return "on" }
        "yes" { return "on" }
        "on" { return "on" }
        "n" { return "off" }
        "no" { return "off" }
        "off" { return "off" }
        default { throw "Invalid sandbox choice: $answer" }
    }
}

function Choose-Messaging {
    $value = Normalize $Messaging
    if (@("none", "telegram", "discord", "both") -contains $value) { return $value }
    if ($DryRun) { return "none" }
    Write-Host "Choose messaging app:"
    Write-Host "  1) None"
    Write-Host "  2) Telegram"
    Write-Host "  3) Discord"
    Write-Host "  4) Both"
    $answer = Read-Host "Messaging [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    switch ($answer.ToLowerInvariant()) {
        "1" { return "none" }
        "none" { return "none" }
        "2" { return "telegram" }
        "telegram" { return "telegram" }
        "3" { return "discord" }
        "discord" { return "discord" }
        "4" { return "both" }
        "both" { return "both" }
        default { throw "Invalid messaging choice: $answer" }
    }
}

function Choose-Memory {
    $value = Normalize $Memory
    if (@("lite", "hybrid") -contains $value) { return $value }
    if ($DryRun) { return "lite" }
    Write-Host "Enable local hybrid memory (ChromaDB + Neo4j + embeddings)?"
    Write-Host "  1) Lite    - fast install, no ML deps (recommended)"
    Write-Host "  2) Hybrid  - installs torch/transformers/chromadb (~hundreds of MB)"
    $answer = Read-Host "Memory [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    switch ($answer.ToLowerInvariant()) {
        "1" { return "lite" }
        "lite" { return "lite" }
        "2" { return "hybrid" }
        "hybrid" { return "hybrid" }
        default { throw "Invalid memory choice: $answer" }
    }
}

function Choose-Model([string]$Title, [string[]]$Models) {
    # $Models[0] is the default. In DryRun (non-interactive) keep the default so
    # automated provider checks stay stable.
    if ($DryRun) { return $Models[0] }
    Write-Host "Choose model for ${Title}:"
    for ($i = 0; $i -lt $Models.Count; $i++) {
        Write-Host ("  {0}) {1}" -f ($i + 1), $Models[$i])
    }
    Write-Host ("  {0}) Other (enter a model string manually)" -f ($Models.Count + 1))
    $answer = Read-Host "Model [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Models[0] }
    $idx = 0
    if ([int]::TryParse($answer, [ref]$idx)) {
        if ($idx -ge 1 -and $idx -le $Models.Count) { return $Models[$idx - 1] }
        if ($idx -eq ($Models.Count + 1)) { return (Read-Host "Model string (LiteLLM format)") }
    }
    throw "Invalid model choice: $answer"
}

function Choose-OpenAIAuth {
    if ($DryRun) { return "apikey" }
    Write-Host "How should the agent authenticate to OpenAI?"
    Write-Host "  1) Paste an OpenAI API key"
    Write-Host "  2) Sign in with ChatGPT (Codex OAuth) - no key; sign in after setup"
    $answer = Read-Host "Auth [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    switch ($answer.ToLowerInvariant()) {
        "1" { return "apikey" }
        "apikey" { return "apikey" }
        "2" { return "oauth" }
        "oauth" { return "oauth" }
        default { throw "Invalid auth choice: $answer" }
    }
}

function Env-Line([string]$Key, [string]$Value = "") {
    if ([string]::IsNullOrEmpty($Value)) { return "$Key=" }
    $escaped = $Value.Replace("\", "\\").Replace('"', '\"').Replace("`r", "").Replace("`n", "")
    return "$Key=""$escaped"""
}

$ProviderChoice = Choose-Provider
$SandboxChoice = Choose-Sandbox
$MessagingChoice = Choose-Messaging
$MemoryChoice = Choose-Memory

$AgentModel = ""
$FastModel = ""
$StrongModel = ""
$MoonshotKey = ""
$MoonshotBase = ""
$OpenRouterKey = ""
$OpenAIKey = ""
$OpenAIBase = ""
$AnthropicKey = ""
$GeminiKey = ""
$OllamaBase = ""
$OpenAIAuthMethod = "apikey"

switch ($ProviderChoice) {
    { $_ -in @("kimi", "moonshot") } {
        $AgentModel = Choose-Model "Kimi / Moonshot" @("moonshot/kimi-k2.6", "moonshot/kimi-k2.5")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $MoonshotKey = Prompt-Secret "MOONSHOT_API_KEY" "Moonshot API key"
        $MoonshotBase = Prompt-Value "Moonshot API base" "https://api.moonshot.ai/v1"
    }
    "ollama" {
        $AgentModel = Choose-Model "Ollama" @("ollama/llama3.2", "ollama/llama3.3:70b", "ollama/qwen2.5:14b")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $OllamaBase = Prompt-Value "Ollama API base, blank for default" ""
    }
    "openrouter" {
        $AgentModel = Choose-Model "OpenRouter" @("openrouter/meta-llama/llama-3.3-70b-instruct", "openrouter/anthropic/claude-sonnet-4-6", "openrouter/anthropic/claude-opus-4-8", "openrouter/openai/gpt-4o")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $OpenRouterKey = Prompt-Secret "OPENROUTER_API_KEY" "OpenRouter API key"
    }
    "openai" {
        $AgentModel = Choose-Model "OpenAI" @("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini", "o3-mini")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $OpenAIAuthMethod = Choose-OpenAIAuth
        if ($OpenAIAuthMethod -eq "apikey") {
            $OpenAIKey = Prompt-Secret "OPENAI_API_KEY" "OpenAI API key"
        }
    }
    "anthropic" {
        $AgentModel = Choose-Model "Anthropic" @("claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-1", "claude-3-5-haiku-latest")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $AnthropicKey = Prompt-Secret "ANTHROPIC_API_KEY" "Anthropic API key"
    }
    "gemini" {
        $AgentModel = Choose-Model "Gemini" @("gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro", "gemini/gemini-2.0-flash", "gemini/gemini-2.0-flash-lite")
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $GeminiKey = Prompt-Secret "GEMINI_API_KEY" "Gemini API key"
    }
    "vllm" {
        $AgentModel = Prompt-Value "vLLM model" "openai/meta-llama/Llama-3.2-8B-Instruct"
        $FastModel = $AgentModel
        $StrongModel = $AgentModel
        $OpenAIKey = Prompt-Value "OpenAI-compatible API key" "dummy"
        $OpenAIBase = Prompt-Value "OpenAI-compatible API base" "http://localhost:8001/v1"
    }
}

$TelegramToken = ""
$TelegramAllowed = ""
$DiscordToken = ""
$DiscordAllowed = ""
if (@("telegram", "both") -contains $MessagingChoice) {
    $TelegramToken = Prompt-Secret "TELEGRAM_BOT_TOKEN" "Telegram bot token"
    $TelegramAllowed = Prompt-Value "Telegram allowed chat IDs, blank = all" ""
}
if (@("discord", "both") -contains $MessagingChoice) {
    $DiscordToken = Prompt-Secret "DISCORD_BOT_TOKEN" "Discord bot token"
    $DiscordAllowed = Prompt-Value "Discord allowed user IDs, blank = all" ""
}

$AgentSandbox = ""
$SandboxFallback = "false"
if ($SandboxChoice -eq "off") { $SandboxFallback = "true" }
$UseHybrid = "false"
if ($MemoryChoice -eq "hybrid") { $UseHybrid = "true" }

function Generated-Env {
    @(
        "# Generated by scripts/install.ps1",
        "# Sandbox mode: $SandboxChoice",
        "# Messaging mode: $MessagingChoice",
        (Env-Line "AGENT_MODEL" $AgentModel),
        (Env-Line "FAST_AGENT_MODEL" $FastModel),
        (Env-Line "STRONG_AGENT_MODEL" $StrongModel),
        (Env-Line "AGENT_ACTION_MAX_REACT_ITERATIONS" "30"),
        (Env-Line "AGENT_MAX_AUTO_CONTINUE_BATCHES" "3"),
        (Env-Line "AGENT_MAX_TOKENS" "2048"),
        (Env-Line "AGENT_PLANNING_MAX_TOKENS" "1024"),
        (Env-Line "AGENT_ARTIFACT_MAX_TOKENS" "20000"),
        (Env-Line "AGENT_FINAL_MAX_TOKENS" "1536"),
        (Env-Line "AGENT_SANDBOX" $AgentSandbox),
        (Env-Line "AGENT_SANDBOX_HOST_FALLBACK" $SandboxFallback),
        (Env-Line "PUBLIC_BASE_URL" "http://localhost:8000"),
        (Env-Line "AGENT_USE_HYBRID_MEMORY" $UseHybrid),
        (Env-Line "MOONSHOT_API_KEY" $MoonshotKey),
        (Env-Line "MOONSHOT_API_BASE" $MoonshotBase),
        (Env-Line "OPENROUTER_API_KEY" $OpenRouterKey),
        (Env-Line "OPENAI_API_KEY" $OpenAIKey),
        (Env-Line "OPENAI_API_BASE" $OpenAIBase),
        (Env-Line "ANTHROPIC_API_KEY" $AnthropicKey),
        (Env-Line "GEMINI_API_KEY" $GeminiKey),
        (Env-Line "OLLAMA_API_BASE" $OllamaBase),
        (Env-Line "TELEGRAM_BOT_TOKEN" $TelegramToken),
        (Env-Line "TELEGRAM_ALLOWED_IDS" $TelegramAllowed),
        (Env-Line "DISCORD_BOT_TOKEN" $DiscordToken),
        (Env-Line "DISCORD_ALLOWED_USER_IDS" $DiscordAllowed)
    )
}

if ($DryRun) {
    Write-Host "# Dry run: generated env for $EnvFile"
    Generated-Env | ForEach-Object { Write-Host $_ }
    exit 0
}

$managed = @{}
@("AGENT_MODEL", "FAST_AGENT_MODEL", "STRONG_AGENT_MODEL", "AGENT_ACTION_MAX_REACT_ITERATIONS", "AGENT_MAX_AUTO_CONTINUE_BATCHES", "AGENT_MAX_TOKENS", "AGENT_PLANNING_MAX_TOKENS", "AGENT_ARTIFACT_MAX_TOKENS", "AGENT_FINAL_MAX_TOKENS", "AGENT_SANDBOX", "AGENT_SANDBOX_HOST_FALLBACK", "PUBLIC_BASE_URL", "AGENT_USE_HYBRID_MEMORY", "MOONSHOT_API_KEY", "MOONSHOT_API_BASE", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_API_BASE", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_BASE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_IDS", "DISCORD_BOT_TOKEN", "DISCORD_ALLOWED_USER_IDS") | ForEach-Object { $managed[$_] = $true }

$out = New-Object System.Collections.Generic.List[string]
if (Test-Path $EnvFile) {
    $backup = "$EnvFile.bak.$(Get-Date -Format yyyyMMddHHmmss)"
    Copy-Item $EnvFile $backup
    Write-Host "Backed up existing env file to $backup"
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match "^([A-Za-z_][A-Za-z0-9_]*)=" -and $managed.ContainsKey($Matches[1])) { continue }
        $out.Add($line)
    }
    if ($out.Count -gt 0) { $out.Add("") }
}
Generated-Env | ForEach-Object { $out.Add($_) }
$parent = Split-Path -Parent $EnvFile
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
Set-Content -Path $EnvFile -Value $out -Encoding UTF8
Write-Host "Wrote $EnvFile"

if (-not $NoStart) {
    if ($SandboxChoice -eq "on") {
        Push-Location $RootDir
        $env:INSTALL_ML = $UseHybrid
        $env:AGENT_USE_HYBRID_MEMORY = $UseHybrid
        try { docker compose up -d --build } finally { Pop-Location }
    }
    else {
        $logs = Join-Path $RootDir "logs"
        New-Item -ItemType Directory -Force -Path $logs | Out-Null
        Push-Location $RootDir
        try {
            py -3 -m venv .run-venv
            $venvPython = Join-Path $RootDir ".run-venv\Scripts\python.exe"
            # The heavy ML stack (requirements-ml.txt) is only installed for hybrid
            # memory mode. --index-strategy unsafe-best-match lets uv pick torch's
            # CPU wheel from the extra index declared in requirements-ml.txt.
            $ReqArgs = @("-r", "requirements.txt")
            if ($MemoryChoice -eq "hybrid") { $ReqArgs += @("-r", "requirements-ml.txt") }
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                # uv resolves and downloads in parallel with a global cache; far
                # faster than pip.
                & uv pip install --index-strategy unsafe-best-match @ReqArgs --python $venvPython
            }
            else {
                # No system uv: bootstrap it into the venv (tiny wheel) so a fresh
                # machine still gets the fast parallel install. Fall back to pip if
                # uv itself can't be installed.
                & $venvPython -m pip install --upgrade pip
                & $venvPython -m pip install uv
                if ($LASTEXITCODE -eq 0) {
                    & $venvPython -m uv pip install --index-strategy unsafe-best-match @ReqArgs --python $venvPython
                }
                else {
                    & $venvPython -m pip install @ReqArgs
                }
            }
            Start-Process -FilePath $venvPython -ArgumentList @("-m", "uvicorn", "gateway:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $RootDir -RedirectStandardOutput (Join-Path $logs "backend-local.stdout.log") -RedirectStandardError (Join-Path $logs "backend-local.stderr.log") -WindowStyle Hidden
            Push-Location (Join-Path $RootDir "control-panel")
            try {
                npm ci --no-audit --no-fund
                Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "npm run dev -- --host 127.0.0.1 --port 5173") -WorkingDirectory (Join-Path $RootDir "control-panel") -RedirectStandardOutput (Join-Path $logs "control-panel-dev.stdout.log") -RedirectStandardError (Join-Path $logs "control-panel-dev.stderr.log") -WindowStyle Hidden
            }
            finally { Pop-Location }
        }
        finally { Pop-Location }
    }
}

Write-Host "Agent AI setup complete."
Write-Host "Control panel: http://localhost:5173"
Write-Host "API health:    http://localhost:8000/health"

if ($OpenAIAuthMethod -eq "oauth") {
    Write-Host ""
    Write-Host "Finish Codex OAuth sign-in (sign in once; the token is stored and refreshed):"
    Write-Host "  - Control panel: Settings -> Authentication -> Sign in with ChatGPT"
    if ($SandboxChoice -eq "off") {
        Write-Host "  - Command line: `$env:PYTHONPATH='src'; .run-venv\Scripts\python.exe -m auth login"
    }
    else {
        Write-Host "  - Command line: docker compose exec -e PYTHONPATH=src agent_core python -m auth login"
        Write-Host "    (In Docker mode the control-panel sign-in is the simplest route.)"
    }
}
