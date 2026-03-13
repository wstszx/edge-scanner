[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ProviderVerificationArgs = @()
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogDir = Join-Path $ScriptDir "data\provider_verification"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "provider_verification_scheduler_$Timestamp.log"
$LatestLogPath = Join-Path $LogDir "provider_verification_scheduler_latest.log"

$Arguments = @("provider_verification.py", "--summary-only") + $ProviderVerificationArgs

& python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

Copy-Item -Path $LogPath -Destination $LatestLogPath -Force
exit $ExitCode
