[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "EdgeScannerProviderVerification",
    [string]$DailyAt = "09:00",
    [string]$Sport = "basketball_nba",
    [string[]]$Providers = @(),
    [switch]$SkipTests,
    [switch]$FailOnAlert
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunnerPath = Join-Path $ScriptDir "scheduled_provider_verification.ps1"
$PowerShellPath = Join-Path $PSHOME "powershell.exe"

if (-not (Test-Path -Path $RunnerPath)) {
    throw "Missing scheduled runner script: $RunnerPath"
}

try {
    $AtTime = [datetime]::Today.Add([timespan]::Parse($DailyAt))
} catch {
    throw "DailyAt must be in HH:mm format."
}

$RunnerArgs = @("--sport", $Sport)

if ($SkipTests) {
    $RunnerArgs += "--skip-tests"
}
if ($FailOnAlert) {
    $RunnerArgs += "--fail-on-alert"
}
if ($Providers.Count -gt 0) {
    $RunnerArgs += "--providers"
    foreach ($Provider in $Providers) {
        $RunnerArgs += $Provider
    }
}

function Quote-TaskToken {
    param([string]$Value)

    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

$TaskCommandParts = @(
    (Quote-TaskToken $PowerShellPath),
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskToken $RunnerPath)
)
$TaskCommandParts += ($RunnerArgs | ForEach-Object { Quote-TaskToken $_ })
$TaskCommand = $TaskCommandParts -join " "

$SchTasksArgs = @(
    "/Create",
    "/F",
    "/SC", "DAILY",
    "/TN", $TaskName,
    "/ST", $DailyAt,
    "/TR", $TaskCommand
)

if ($PSCmdlet.ShouldProcess($TaskName, "Register or update scheduled task")) {
    & schtasks.exe @SchTasksArgs
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks.exe failed with exit code $LASTEXITCODE."
    }
    Write-Output "Registered scheduled task '$TaskName' at $DailyAt."
} else {
    Write-Output ("schtasks.exe " + (($SchTasksArgs | ForEach-Object { Quote-TaskToken $_ }) -join " "))
}
