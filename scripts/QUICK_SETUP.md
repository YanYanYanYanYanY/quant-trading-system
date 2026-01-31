# Quick Setup Guide - Daily Stock Updates

## 🚀 Fastest Setup (5 minutes)

### Step 1: Set Environment Variable

**Windows PowerShell (as Administrator):**
```powershell
[System.Environment]::SetEnvironmentVariable('POLYGON_API_KEY', 'your_api_key_here', 'User')
```

**Or manually:**
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to **Advanced** tab → **Environment Variables**
3. Under **User variables**, click **New**
4. Variable name: `POLYGON_API_KEY`
5. Variable value: `your_polygon_api_key_here`
6. Click **OK**

### Step 2: Test the Script

```powershell
# Test the update script
cd C:\Users\qqyan\quant_trading
.\scripts\update_stocks_daily.ps1
```

Check the log file: `logs/daily_update_YYYY-MM-DD.log`

### Step 3: Schedule with Task Scheduler

1. Open **Task Scheduler** (search in Windows)
2. Click **Create Basic Task**
3. Fill in:
   - **Name**: `Daily Stock Update`
   - **Trigger**: Daily at 6:00 PM (or your preferred time)
   - **Action**: Start a program
   - **Program**: `powershell.exe`
   - **Arguments**: 
     ```
     -ExecutionPolicy Bypass -File "C:\Users\qqyan\quant_trading\scripts\update_stocks_daily.ps1"
     ```
   - **Start in**: `C:\Users\qqyan\quant_trading`

4. Click **Finish**

### Step 4: Test the Scheduled Task

1. Right-click the task → **Run**
2. Check `logs/daily_update_YYYY-MM-DD.log` for results
3. Verify files in `engine/data/raw/stocks/` are updated

## ✅ Done!

Your stock data will now update automatically every day at the scheduled time.

## 📋 What Gets Updated

- All tickers from `engine/data/raw/stocks/good_quality_stock_tickers_200.txt`
- Files saved as: `{SYMBOL}_1d.csv` and `{SYMBOL}_1d.parquet`
- Location: `engine/data/raw/stocks/`
- Logs: `logs/daily_update_YYYY-MM-DD.log`

## 🔍 Verify It's Working

1. **Check logs**: `logs/daily_update_*.log`
2. **Check file dates**: Files in `engine/data/raw/stocks/` should have recent modification dates
3. **Check Task Scheduler**: Task should show "Last Run Result: 0x0" (success)

## ❓ Need Help?

See `scripts/README_SCHEDULER.md` for detailed troubleshooting and alternative setup methods.
