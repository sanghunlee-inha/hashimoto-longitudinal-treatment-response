$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# PowerShell 5.1 + native Python/conda: force UTF-8 for readable Korean paths.
try { chcp 65001 | Out-Null } catch {}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Desktop = [Environment]::GetFolderPath("Desktop")
$Downloads = Join-Path $env:USERPROFILE "Downloads"
$ScriptName = "T1D_ADDS_V35_4_EXACT_LOCKED_BENCHMARK_DECOUPLING.py"

Write-Host "====================================================================================================" -ForegroundColor Cyan
Write-Host " V35.4 EXACT-LOCKED CLINICAL BENCHMARK + TRAJECTORY DECOUPLING" -ForegroundColor Cyan
Write-Host "====================================================================================================" -ForegroundColor Cyan

$Roots = @(Get-ChildItem -LiteralPath $Desktop -Directory -ErrorAction Stop | Where-Object { $_.Name -like "26.09.11*ADDS" })
if($Roots.Count -eq 0){ throw "PROJECT_ROOT_NOT_FOUND" }
$ROOT = ($Roots | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
Write-Host "ROOT=$ROOT" -ForegroundColor Green

$ScriptCandidates=@()
$ScriptCandidates += @(Get-ChildItem -LiteralPath $Downloads -Filter $ScriptName -File -ErrorAction SilentlyContinue)
$ScriptCandidates += @(Get-ChildItem -LiteralPath $ROOT -Filter $ScriptName -File -Recurse -ErrorAction SilentlyContinue)
$SCRIPT = $ScriptCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if($null -eq $SCRIPT){ throw "V35_4_MASTER_NOT_FOUND=$ScriptName" }
Write-Host "SCRIPT=$($SCRIPT.FullName)" -ForegroundColor Green

$Conda = Get-Command conda.exe -ErrorAction SilentlyContinue
if($null -eq $Conda){ throw "CONDA_NOT_FOUND" }

powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
powercfg /change monitor-timeout-ac 0 | Out-Null

$LogDir = Join-Path $ROOT "V35_4_LAUNCH_LOGS"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LogDir ("V35_4_"+$TS+".log")
Write-Host "LOG=$LOG"

# IMPORTANT: Windows PowerShell 5.1 wraps native STDERR as NativeCommandError.
# Python/scikit-learn warnings use STDERR but are not failures. Temporarily use Continue
# so warnings are logged while the true process exit code remains authoritative.
$SavedEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & conda.exe run --no-capture-output -n t1d_tads_v13 python "$($SCRIPT.FullName)" --root "$ROOT" --bootstrap 20000 --workers 20 --seed 42 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $LOG
    $RC = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $SavedEAP
}

Write-Host ""
Write-Host "EXIT_CODE=$RC"
if($RC -ne 0){
    Write-Host "====================================================================================================" -ForegroundColor Red
    Write-Host " LAST 160 LOG LINES" -ForegroundColor Red
    Write-Host "====================================================================================================" -ForegroundColor Red
    Get-Content -LiteralPath $LOG -Tail 160
    throw "V35.4 FAILED. Full log: $LOG"
}
Write-Host "FINAL_STATUS=PASS_V35_4" -ForegroundColor Green
Write-Host "LOG=$LOG" -ForegroundColor Green
