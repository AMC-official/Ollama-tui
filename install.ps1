# Installs ollama-tui on Windows.
#
# Works from a source checkout (builds every .go file in this folder) and from
# a release archive (installs the ready-made ollama-tui.exe).
#
#   .\install.ps1              installs to %USERPROFILE%\bin
#   .\install.ps1 -AddToPath   also adds that folder to your user PATH

param([switch]$AddToPath)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$exe = "ollama-tui.exe"

# 1. Get a binary: build it from source, or use the one from the release archive.
if ((Test-Path "go.mod") -and (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Host "Source code detected. Building ollama-tui (all the .go files in this folder)..." -ForegroundColor Cyan
    # -s -w drops debug info and -trimpath drops build paths: a smaller binary.
    go build -trimpath -ldflags "-s -w" -o $exe .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: the build failed." -ForegroundColor Red
        exit 1
    }
} elseif (Test-Path $exe) {
    Write-Host "Pre-compiled binary detected." -ForegroundColor Cyan
} elseif (Test-Path "go.mod") {
    Write-Host "Error: this is the source code, but Go is not installed (see https://go.dev/dl/)." -ForegroundColor Red
    exit 1
} else {
    Write-Host "Error: found neither the source code (go.mod) nor $exe in this folder." -ForegroundColor Red
    exit 1
}

# 2. Copy it to a folder in your profile.
$installPath = Join-Path $env:USERPROFILE "bin"
if (!(Test-Path -Path $installPath)) {
    New-Item -ItemType Directory -Path $installPath | Out-Null
}
Copy-Item -Path $exe -Destination (Join-Path $installPath $exe) -Force

# 3. Make sure it can be found.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
$onPath = ($userPath -split ";") -contains $installPath

if (-not $onPath -and $AddToPath) {
    $newPath = ($userPath.TrimEnd(";") + ";" + $installPath).TrimStart(";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "Added $installPath to your user PATH. Open a new terminal to use it." -ForegroundColor Green
} elseif (-not $onPath) {
    Write-Host "Note: $installPath is not in your PATH. Run this script again with -AddToPath, or add it yourself." -ForegroundColor Yellow
}

Write-Host "Done! Run 'ollama-tui' in a new terminal." -ForegroundColor Green
