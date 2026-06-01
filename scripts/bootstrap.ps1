param(
    [string]$RepoUrl = $env:AGENT_BOOTSTRAP_REPO_URL,
    [string]$Branch = $env:AGENT_BOOTSTRAP_BRANCH,
    [string]$InstallDir = $env:AGENT_BOOTSTRAP_INSTALL_DIR,
    [string]$Provider = $env:AGENT_INSTALL_PROVIDER,
    [string]$Sandbox = $env:AGENT_INSTALL_SANDBOX,
    [string]$Messaging = $env:AGENT_INSTALL_MESSAGING,
    [string]$Memory = $env:AGENT_INSTALL_MEMORY,
    [switch]$DryRun,
    [switch]$NoStart,
    [switch]$NoUpdate
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$DefaultRepoUrl = "https://github.com/Aspct3434/agent-ai.git"
if ([string]::IsNullOrWhiteSpace($RepoUrl)) { $RepoUrl = $DefaultRepoUrl }
if ([string]::IsNullOrWhiteSpace($Branch)) { $Branch = "master" }
if ([string]::IsNullOrWhiteSpace($InstallDir)) { $InstallDir = Join-Path $HOME "distill" }

function Step([string]$Message) { Write-Host "==> $Message" }
function Has-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

function Run-Cmd([string]$Description, [string]$File, [string[]]$CommandArgs) {
    Step $Description
    if ($DryRun) {
        Write-Host ("[dry-run] " + $File + " " + ($CommandArgs -join " "))
        return
    }
    & $File @CommandArgs
}

function Ensure-Git {
    if (Has-Command "git") { Step "Git is available"; return }
    if (Has-Command "winget") {
        Run-Cmd "Install Git with winget" "winget" @("install", "--id", "Git.Git", "-e", "--source", "winget")
        return
    }
    throw "Git is required. Install Git for Windows, reopen PowerShell, then rerun this bootstrap."
}

function Ensure-Docker {
    if ($Sandbox -eq "off") { Step "Sandbox off selected; Docker is not required"; return }
    if (Has-Command "docker") {
        & docker compose version *> $null
        $composeOk = $LASTEXITCODE -eq 0
        & docker info *> $null
        $running = $LASTEXITCODE -eq 0
        if ($composeOk -and $running) { Step "Docker Compose is available and Docker is running"; return }
        if ($composeOk -and -not $running) {
            Write-Host "Docker is installed, but Docker Desktop is not running. Start it, then rerun this bootstrap."
            if (-not $DryRun) { throw "Docker is not running." }
            return
        }
    }
    if (Has-Command "winget") {
        Run-Cmd "Install Docker Desktop with winget" "winget" @("install", "--id", "Docker.DockerDesktop", "-e", "--source", "winget")
    }
    Write-Host "Docker Desktop must be installed and running before sandbox-on startup can finish."
    if (-not $DryRun) { throw "Docker Compose is not available yet." }
}

function Is-AgentRepo([string]$Path) {
    return ((Test-Path (Join-Path $Path "scripts\install.ps1")) -and (Test-Path (Join-Path $Path "docker-compose.yml")) -and (Test-Path (Join-Path $Path "src\gateway.py")))
}

function Ensure-Repo {
    if (Test-Path $InstallDir) {
        if (Is-AgentRepo $InstallDir) {
            Step "Found existing Distill checkout at $InstallDir"
            if (-not $NoUpdate) {
                Run-Cmd "Fetch latest repo changes" "git" @("-C", $InstallDir, "fetch", "origin", $Branch)
                Run-Cmd "Checkout selected branch" "git" @("-C", $InstallDir, "checkout", $Branch)
                Run-Cmd "Pull selected branch" "git" @("-C", $InstallDir, "pull", "--ff-only", "origin", $Branch)
            }
            return
        }
        $items = Get-ChildItem -Force -Path $InstallDir -ErrorAction SilentlyContinue
        if ($items.Count -gt 0) { throw "InstallDir exists but is not a Distill checkout: $InstallDir" }
    }
    else {
        $parent = Split-Path -Parent $InstallDir
        if ($DryRun) { Write-Host "[dry-run] New-Item -ItemType Directory -Force -Path $parent" }
        else { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    }
    Run-Cmd "Clone Distill repository" "git" @("clone", "--depth", "1", "--single-branch", "--branch", $Branch, $RepoUrl, $InstallDir)
}

function Run-Installer {
    $installer = Join-Path $InstallDir "scripts\install.ps1"
    $args = New-Object System.Collections.Generic.List[string]
    $args.Add("-ExecutionPolicy"); $args.Add("Bypass"); $args.Add("-File"); $args.Add($installer)
    if ($Provider) { $args.Add("-Provider"); $args.Add($Provider) }
    if ($Sandbox) { $args.Add("-Sandbox"); $args.Add($Sandbox) }
    if ($Messaging) { $args.Add("-Messaging"); $args.Add($Messaging) }
    if ($Memory) { $args.Add("-Memory"); $args.Add($Memory) }
    if ($DryRun) { $args.Add("-DryRun") }
    if ($NoStart) { $args.Add("-NoStart") }
    Run-Cmd "Run Distill installer" "powershell" $args.ToArray()
}

Write-Host "Distill bootstrap installer"
Write-Host "Install directory: $InstallDir"
Write-Host "Repository:        $RepoUrl"
Write-Host "Branch:            $Branch"
Write-Host ""

Ensure-Git
Ensure-Docker
Ensure-Repo
Run-Installer
