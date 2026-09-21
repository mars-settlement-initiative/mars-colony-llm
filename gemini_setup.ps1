# Run from PowerShell: .\gemini_setup.ps1
# The key stays in this process environment and is not written to a file.
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Gemini API key from a Free Tier AI Studio project' -AsSecureString)).Password
$env:MARS_LLM_MODE = 'gemini'
$env:GEMINI_MODEL = 'gemini-3.1-flash-lite'
$geminiPython = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $geminiPython) {
    & $geminiPython (Join-Path $PSScriptRoot 'run.py') --mode gemini
} else {
    python (Join-Path $PSScriptRoot 'run.py') --mode gemini
}
