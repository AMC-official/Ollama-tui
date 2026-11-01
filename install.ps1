Write-Host "Building ollama-tui..." -ForegroundColor Cyan
go build -o ollama-tui.exe main.go
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit
}

$installPath = "$env:USERPROFILE\bin"
if (!(Test-Path -Path $installPath)) {
    New-Item -ItemType Directory -Path $installPath | Out-Null
}

Move-Item -Path "ollama-tui.exe" -Destination "$installPath\ollama-tui.exe" -Force
Write-Host "Done! Run 'ollama-tui.exe' (Make sure $installPath is in your PATH)" -ForegroundColor Green
