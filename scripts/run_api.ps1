# Run the Quant Trading API
# Run from project root. Requires: pip install uvicorn (or use venv with full deps)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ApiDir = Join-Path $ProjectRoot "api"

Set-Location $ApiDir
$env:PYTHONPATH = "$ProjectRoot;$ApiDir"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
