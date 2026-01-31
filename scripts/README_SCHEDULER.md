# Daily Stock Update Scheduler Setup

This guide explains how to set up automatic daily updates for OHLCV stock data.

## Options

### Option 1: Windows Task Scheduler (Recommended for Windows)

Windows Task Scheduler is built into Windows and can run scripts automatically.

#### Step 1: Create the Scheduled Task

1. Open **Task Scheduler** (search for "Task Scheduler" in Windows)
2. Click **Create Basic Task** (or **Create Task** for more options)
3. Configure the task:
   - **Name**: `Daily Stock Data Update`
   - **Description**: `Automatically update daily stock aggregates`
   - **Trigger**: Daily, at your preferred time (e.g., 6:00 PM)
   - **Action**: Start a program
   - **Program/script**: `powershell.exe`
   - **Arguments**: 
     ```
     -ExecutionPolicy Bypass -File "C:\Users\qqyan\quant_trading\scripts\update_stocks_daily.ps1"
     ```
   - **Start in**: `C:\Users\qqyan\quant_trading`

#### Step 2: Set Environment Variable

Before the task runs, set the `POLYGON_API_KEY` environment variable:

**Option A: System-wide (recommended)**
1. Open **System Properties** → **Environment Variables**
2. Under **System variables**, click **New**
3. Variable name: `POLYGON_API_KEY`
4. Variable value: `your_polygon_api_key_here`
5. Click **OK** to save

**Option B: User-specific**
1. Same as above, but add to **User variables** instead

**Option C: In Task Scheduler**
1. In Task Scheduler, edit your task
2. Go to **Actions** tab → Edit action
3. Add environment variable in **Add arguments** or use **Start in** with a batch file that sets it

#### Step 3: Test the Task

1. Right-click your task → **Run**
2. Check the log file in `logs/daily_update_YYYY-MM-DD.log`
3. Verify data files are updated in `engine/data/raw/stocks/`

#### Step 4: Configure Task Settings

In Task Scheduler, go to your task's **Settings** tab:
- ✅ **Allow task to be run on demand**
- ✅ **Run task as soon as possible after a scheduled start is missed**
- ✅ **If the task fails, restart every**: 1 hour (optional)
- ✅ **Stop the task if it runs longer than**: 2 hours (optional)

---

### Option 2: Python Scheduler Script

Run a Python script that schedules updates internally.

#### Setup

```bash
# Set environment variable
set POLYGON_API_KEY=your_api_key_here

# Run scheduler (updates daily at 6 PM)
python scripts/scheduler.py --time 18:00 --service
```

#### Run as Windows Service

To run as a Windows service, you can use **NSSM** (Non-Sucking Service Manager):

1. Download NSSM from https://nssm.cc/download
2. Install the service:
   ```cmd
   nssm install DailyStockUpdate "C:\Users\qqyan\quant_trading\.venv\Scripts\python.exe" "C:\Users\qqyan\quant_trading\scripts\scheduler.py --time 18:00 --service"
   ```
3. Set environment variable in NSSM:
   - Open NSSM GUI: `nssm edit DailyStockUpdate`
   - Go to **Environment** tab
   - Add: `POLYGON_API_KEY=your_api_key_here`
4. Start the service:
   ```cmd
   nssm start DailyStockUpdate
   ```

---

### Option 3: Batch File (Simple)

Use the provided batch file with Task Scheduler:

1. In Task Scheduler, set:
   - **Program/script**: `C:\Users\qqyan\quant_trading\scripts\update_stocks_daily.bat`
   - **Start in**: `C:\Users\qqyan\quant_trading`

2. Set `POLYGON_API_KEY` environment variable (see Option 1, Step 2)

---

## Verification

After setup, verify it's working:

1. **Check logs**: Look in `logs/daily_update_YYYY-MM-DD.log`
2. **Check data files**: Verify `engine/data/raw/stocks/{SYMBOL}_1d.csv` and `.parquet` files are updating
3. **Check file timestamps**: Files should have recent modification dates

## Troubleshooting

### Task doesn't run
- Check Task Scheduler **History** tab for errors
- Verify Python path is correct
- Check that `POLYGON_API_KEY` is set correctly
- Run the script manually to test: `python -m engine.data.update_daily_aggregates`

### Permission errors
- Ensure the task runs with appropriate user permissions
- Check that the virtual environment is accessible
- Verify file write permissions in `engine/data/raw/stocks/`

### API errors
- Verify your Polygon API key is valid
- Check API rate limits (script includes delays)
- Review log files for specific error messages

### Script not found
- Verify paths are absolute or relative to project root
- Check that `engine` module is installed or on PYTHONPATH

## Manual Run

To run manually at any time:

```bash
# Using PowerShell
.\scripts\update_stocks_daily.ps1

# Using batch file
scripts\update_stocks_daily.bat

# Using Python directly
python -m engine.data.update_daily_aggregates
```

## Recommended Schedule

- **Time**: 6:00 PM - 7:00 PM (after market close)
- **Frequency**: Daily (Monday-Friday, or every day if you want weekend data)
- **Note**: Markets are closed on weekends, so updates on Saturday/Sunday may return no new data (this is normal)
