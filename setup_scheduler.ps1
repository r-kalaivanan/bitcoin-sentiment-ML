# Bitcoin Sentiment ML - PowerShell Task Scheduler Setup
# This script creates scheduled tasks for automated daily workflow

Write-Host "🚀 Bitcoin Sentiment ML - Task Scheduler Setup" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green

$ScriptPath = $PSScriptRoot
$PythonPath = Join-Path $ScriptPath "venv\Scripts\python.exe"
$ProjectPath = $ScriptPath

Write-Host "Current directory: $ProjectPath" -ForegroundColor Cyan
Write-Host "Python path: $PythonPath" -ForegroundColor Cyan

# Check if Python exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Python not found at $PythonPath" -ForegroundColor Red
    Write-Host "Please make sure your virtual environment is set up correctly" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Python found at $PythonPath" -ForegroundColor Green

# Function to create scheduled task
function Create-BitcoinTask {
    param(
        [string]$TaskName,
        [string]$Description,
        [string]$Arguments,
        [string]$Schedule,
        [string]$StartTime = "09:00"
    )
    
    try {
        # Create action
        $Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $Arguments -WorkingDirectory $ProjectPath
        
        # Create trigger based on schedule
        switch ($Schedule) {
            "Daily" { $Trigger = New-ScheduledTaskTrigger -Daily -At $StartTime }
            "Hourly" { $Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365) }
            "Startup" { $Trigger = New-ScheduledTaskTrigger -AtStartup }
        }
        
        # Create settings
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        
        # Register task
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description $Description -Force
        
        Write-Host "✅ $TaskName created successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create $TaskName : $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

Write-Host ""
Write-Host "📅 Creating daily workflow task..." -ForegroundColor Yellow
$DailySuccess = Create-BitcoinTask -TaskName "Bitcoin-ML-Daily-Workflow" -Description "Bitcoin ML Daily Analysis and Prediction" -Arguments "`"$ProjectPath\scripts\daily_workflow_manager.py`" daily" -Schedule "Daily" -StartTime "09:00"

Write-Host ""
Write-Host "⏰ Creating hourly quick update task..." -ForegroundColor Yellow
$HourlySuccess = Create-BitcoinTask -TaskName "Bitcoin-ML-Quick-Update" -Description "Bitcoin ML Quick Sentiment and Price Update" -Arguments "`"$ProjectPath\scripts\daily_workflow_manager.py`" quick" -Schedule "Hourly"

Write-Host ""
$RealtimeChoice = Read-Host "🔴 Do you want to create a real-time data streaming task? (y/n)"
$RealtimeSuccess = $false

if ($RealtimeChoice -eq 'y' -or $RealtimeChoice -eq 'Y') {
    Write-Host "🔴 Creating real-time data streaming task..." -ForegroundColor Yellow
    $RealtimeSuccess = Create-BitcoinTask -TaskName "Bitcoin-ML-Realtime-Data" -Description "Bitcoin ML Real-time Price Data Streaming" -Arguments "`"$ProjectPath\scripts\realtime_data_streamer.py`"" -Schedule "Startup"
    
    if ($RealtimeSuccess) {
        Write-Host "⚠️ Real-time task will run continuously - you can disable it later if needed" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "📊 Task Summary:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan

if ($DailySuccess) {
    Write-Host "✅ Bitcoin-ML-Daily-Workflow   - Daily at 9:00 AM" -ForegroundColor Green
} else {
    Write-Host "❌ Bitcoin-ML-Daily-Workflow   - Failed to create" -ForegroundColor Red
}

if ($HourlySuccess) {
    Write-Host "✅ Bitcoin-ML-Quick-Update     - Every hour" -ForegroundColor Green
} else {
    Write-Host "❌ Bitcoin-ML-Quick-Update     - Failed to create" -ForegroundColor Red
}

if ($RealtimeSuccess) {
    Write-Host "✅ Bitcoin-ML-Realtime-Data    - At startup" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔧 To manage these tasks:" -ForegroundColor Cyan
Write-Host "- Open Task Scheduler (taskschd.msc)" -ForegroundColor White
Write-Host "- Look for tasks starting with 'Bitcoin-ML-'" -ForegroundColor White
Write-Host "- You can enable/disable/modify them as needed" -ForegroundColor White

Write-Host ""
Write-Host "🎯 Testing daily workflow task..." -ForegroundColor Yellow

try {
    Start-ScheduledTask -TaskName "Bitcoin-ML-Daily-Workflow"
    Write-Host "✅ Daily workflow task started successfully!" -ForegroundColor Green
    Write-Host "Check the logs/ directory for output" -ForegroundColor Cyan
}
catch {
    Write-Host "⚠️ Could not start task immediately, but it's scheduled for 9:00 AM daily" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📚 Next steps:" -ForegroundColor Cyan
Write-Host "1. 🔧 Set up Reddit API for better sentiment data (see API_SETUP_GUIDE.md)" -ForegroundColor White
Write-Host "2. 📊 Monitor logs in the logs/ directory" -ForegroundColor White
Write-Host "3. 🎯 View predictions in the predictions/ directory" -ForegroundColor White
Write-Host "4. 📈 Run dashboard: python scripts/enhanced_dashboard.py" -ForegroundColor White

Write-Host ""
Write-Host "🎉 Setup complete! Your Bitcoin ML system will now run automatically." -ForegroundColor Green

Read-Host "Press Enter to exit"