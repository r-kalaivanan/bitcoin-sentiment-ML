# Bitcoin Sentiment ML - Simple Task Scheduler Setup
Write-Host "Setting up Bitcoin ML automated tasks..." -ForegroundColor Green

$PythonPath = "C:/Users/prema/Desktop/Projects/bitcoin-sentiment-ml/venv/Scripts/python.exe"
$ProjectPath = "C:/Users/prema/Desktop/Projects/bitcoin-sentiment-ml"

# Daily workflow task
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ProjectPath\scripts\daily_workflow_manager.py`" daily" -WorkingDirectory $ProjectPath
$Trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "Bitcoin-ML-Daily" -Action $Action -Trigger $Trigger -Settings $Settings -Force

Write-Host "Daily task created - runs at 9:00 AM" -ForegroundColor Green

# Quick update task  
$Action2 = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ProjectPath\scripts\daily_workflow_manager.py`" quick" -WorkingDirectory $ProjectPath
$Trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365)
Register-ScheduledTask -TaskName "Bitcoin-ML-Hourly" -Action $Action2 -Trigger $Trigger2 -Settings $Settings -Force

Write-Host "Hourly task created - runs every hour" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green