$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$candidates = @(
    ".venv\Scripts\streamlit.exe",
    ".venv-1\Scripts\streamlit.exe"
)

foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        & $candidate run app.py
        exit $LASTEXITCODE
    }
}

Write-Host "Streamlit wurde in .venv oder .venv-1 nicht gefunden. Bitte zuerst die Abhängigkeiten installieren."
exit 1
