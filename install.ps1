$ErrorActionPreference = "Stop"

if (Test-Path "main.go") {
    Write-Host "Source code detected. Building ollama-tui..." -ForegroundColor Cyan
    go build -o ollama-tui.exe main.go
} elseif (Test-Path "ollama-tui.exe") {
    Write-Host "Pre-compiled binary detected." -ForegroundColor Cyan
} else {
    Write-Host "Error: Neither main.go nor ollama-tui.exe found." -ForegroundColor Red
    exit 1
}

$installPath = "$env:USERPROFILE\bin"
if (!(Test-Path -Path $installPath)) {
    New-Item -ItemType Directory -Path $installPath | Out-Null
}

Move-Item -Path "ollama-tui.exe" -Destination "$installPath\ollama-tui.exe" -Force
Write-Host "Done! Run 'ollama-tui.exe' (Make sure $installPath is in your PATH)" -ForegroundColor Green
