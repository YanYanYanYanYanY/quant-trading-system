# PowerShell script to run daily stock updates
# This script can be scheduled with Windows Task Scheduler

param(
    [string]$ApiKey = $null
)

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Set working directory
Set-Location $ProjectRoot

# Activate virtual environment
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython"
    exit 1
}

# Set API key from environment if not provided
if (-not $ApiKey) {
    $ApiKey = $env:POLYGON_API_KEY
}

# Log file
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogFile = Join-Path $LogDir "daily_update_$(Get-Date -Format 'yyyy-MM-dd').log"

# Function to write log
function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

Write-Log "Starting daily stock update..."
Write-Log "Project root: $ProjectRoot"
Write-Log "Python: $VenvPython"

# Run the update script
try {
    if ($ApiKey) {
        Write-Log "Using API key from parameter/environment"
        & $VenvPython -m engine.data.update_daily_aggregates $ApiKey 2>&1 | Tee-Object -FilePath $LogFile -Append
    } else {
        Write-Log "Using API key from POLYGON_API_KEY environment variable"
        & $VenvPython -m engine.data.update_daily_aggregates 2>&1 | Tee-Object -FilePath $LogFile -Append
    }
    
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        Write-Log "Daily update completed successfully"
        exit 0
    } else {
        Write-Log "Daily update completed with errors (exit code: $ExitCode)"
        exit $ExitCode
    }
} catch {
    Write-Log "Error running daily update: $_"
    exit 1
}
