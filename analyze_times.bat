@echo off
echo Running execution time analysis...
powershell -ExecutionPolicy Bypass -File "analyze_execution_times.ps1" %*
pause