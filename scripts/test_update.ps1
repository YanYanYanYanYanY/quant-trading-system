# Test script to verify daily update setup
# Run this to test your configuration before scheduling

Write-Host "Testing Daily Stock Update Configuration..." -ForegroundColor Cyan
Write-Host ""

# Check project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Write-Host "Project root: $ProjectRoot" -ForegroundColor Green

# Check virtual environment
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "✓ Virtual environment found: $VenvPython" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment NOT found: $VenvPython" -ForegroundColor Red
    exit 1
}

# Check API key
$ApiKey = $env:POLYGON_API_KEY
if ($ApiKey) {
    Write-Host "✓ POLYGON_API_KEY environment variable is set" -ForegroundColor Green
} else {
    Write-Host "✗ POLYGON_API_KEY environment variable is NOT set" -ForegroundColor Red
    Write-Host "  Set it with: [System.Environment]::SetEnvironmentVariable('POLYGON_API_KEY', 'your_key', 'User')" -ForegroundColor Yellow
    exit 1
}

# Check ticker file
$TickerFile = Join-Path $ProjectRoot "engine\data\raw\stocks\good_quality_stock_tickers_200.txt"
if (Test-Path $TickerFile) {
    $TickerCount = (Get-Content $TickerFile | Where-Object { $_.Trim() -ne "" }).Count
    Write-Host "✓ Ticker file found: $TickerCount tickers" -ForegroundColor Green
} else {
    Write-Host "✗ Ticker file NOT found: $TickerFile" -ForegroundColor Red
    exit 1
}

# Check data directory
$DataDir = Join-Path $ProjectRoot "engine\data\raw\stocks"
if (Test-Path $DataDir) {
    Write-Host "✓ Data directory exists: $DataDir" -ForegroundColor Green
} else {
    Write-Host "⚠ Data directory does not exist, will be created: $DataDir" -ForegroundColor Yellow
}

# Check logs directory
$LogDir = Join-Path $ProjectRoot "logs"
if (Test-Path $LogDir) {
    Write-Host "✓ Logs directory exists: $LogDir" -ForegroundColor Green
} else {
    Write-Host "⚠ Logs directory does not exist, will be created: $LogDir" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Configuration looks good! Running test update..." -ForegroundColor Cyan
Write-Host ""

# Run a test update (just check one ticker or dry run)
# For now, just verify the script can be imported
try {
    $TestResult = & $VenvPython -c "from engine.data.update_daily_aggregates import update_all_daily_aggregates; print('Import successful')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python module imports successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Python module import failed:" -ForegroundColor Red
        Write-Host $TestResult
        exit 1
    }
} catch {
    Write-Host "✗ Error testing Python module: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ All checks passed! You can now schedule the daily update." -ForegroundColor Green
Write-Host ""
Write-Host "To test a full update (downloads data for all tickers):" -ForegroundColor Cyan
Write-Host "  .\scripts\update_stocks_daily.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "To schedule with Task Scheduler, see: scripts\QUICK_SETUP.md" -ForegroundColor Cyan
