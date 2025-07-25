# PowerShell script to analyze execution times from CSV file
# Usage: .\analyze_execution_times.ps1 [csv_file_path] [top_count]

param(
    [string]$CsvFile = "execution_times2.csv",
    [int]$TopCount = 50
)

# Check if file exists
if (-not (Test-Path $CsvFile)) {
    Write-Error "File '$CsvFile' not found!"
    exit 1
}

Write-Host "Analyzing execution times from: $CsvFile" -ForegroundColor Cyan
Write-Host "Looking for top $TopCount operations..." -ForegroundColor Cyan
Write-Host ""

# Read the CSV file and parse execution times
$data = Get-Content $CsvFile | ForEach-Object {
    $parts = $_ -split ','
    if ($parts.Length -eq 3) {
        $name = $parts[0]
        $op = $parts[1]
        $timeStr = $parts[2]
        
        # Convert time to seconds for sorting
        $timeInSeconds = 0
        if ($timeStr -match '(\d+\.?\d*)s$') {
            $timeInSeconds = [double]$matches[1]
        } elseif ($timeStr -match '(\d+\.?\d*)ms$') {
            $timeInSeconds = [double]$matches[1] / 1000
        } elseif ($timeStr -match '(\d+\.?\d*)µs$') {
            $timeInSeconds = [double]$matches[1] / 1000000
        }
        
        [PSCustomObject]@{
            Name = $name
            Operation = $op
            TimeString = $timeStr
            TimeInSeconds = $timeInSeconds
        }
    }
}

# Sort by time in descending order and take top N
$topOperations = $data | Sort-Object TimeInSeconds -Descending | Select-Object -First $TopCount

# Display results
Write-Host "Top $TopCount Most Time-Consuming Operations:" -ForegroundColor Green
Write-Host ("=" * 50) -ForegroundColor Green
$rank = 1
$totalTime = 0

foreach ($item in $topOperations) {
    $totalTime += $item.TimeInSeconds
    
    # Convert seconds to human-readable format
    $timeDisplay = $item.TimeString
    if ($item.TimeInSeconds -gt 60) {
        $minutes = [math]::Floor($item.TimeInSeconds / 60)
        $seconds = $item.TimeInSeconds % 60
        $timeDisplay += " (~$minutes min $([math]::Round($seconds, 1))s)"
    }
    
    Write-Host "$rank. " -NoNewline -ForegroundColor White
    Write-Host "$($item.Name) " -NoNewline -ForegroundColor Yellow
    Write-Host "[$($item.Operation)] " -NoNewline -ForegroundColor Magenta
    Write-Host "- $timeDisplay" -ForegroundColor Cyan
    $rank++
}

Write-Host ""
Write-Host "Summary:" -ForegroundColor Green
Write-Host "--------" -ForegroundColor Green
Write-Host "Total operations analyzed: $($data.Count)" -ForegroundColor White
Write-Host "Top $TopCount operations total time: $([math]::Round($totalTime, 2)) seconds" -ForegroundColor White
Write-Host "Top $TopCount operations total time: $([math]::Round($totalTime / 60, 2)) minutes" -ForegroundColor White
Write-Host "Top $TopCount operations total time: $([math]::Round($totalTime / 3600, 2)) hours" -ForegroundColor White

# Operation type analysis
Write-Host ""
Write-Host "Operation Type Breakdown (Top $TopCount):" -ForegroundColor Green
Write-Host "----------------------------------------" -ForegroundColor Green
$opTypes = $topOperations | Group-Object Operation | Sort-Object Count -Descending
foreach ($opType in $opTypes) {
    $opTotalTime = ($topOperations | Where-Object Operation -eq $opType.Name | Measure-Object TimeInSeconds -Sum).Sum
    Write-Host "$($opType.Name): $($opType.Count) operations, $([math]::Round($opTotalTime, 2))s total" -ForegroundColor Cyan
}