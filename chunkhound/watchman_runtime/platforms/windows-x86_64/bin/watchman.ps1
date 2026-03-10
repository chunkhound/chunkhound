param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$version = "0.0.0-chunkhound-sidecar-placeholder"

function Write-Stderr([string]$Message, [int]$Code) {
    [Console]::Error.WriteLine($Message)
    exit $Code
}

function Emit-Json($Payload) {
    [Console]::Out.WriteLine(($Payload | ConvertTo-Json -Compress -Depth 10))
    [Console]::Out.Flush()
}

if ($RemainingArgs.Count -gt 0 -and $RemainingArgs[0] -eq "--version") {
    [Console]::Out.WriteLine("watchman $version")
    exit 0
}

$foreground = $false
$persistent = $false
$jsonCommand = $false
$noSpawn = $false
$noPretty = $false
$sockname = $null
$statefile = $null
$logfile = $null
$serverEncoding = $null
$outputEncoding = $null

for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
    $arg = $RemainingArgs[$i]
    switch ($arg) {
        "--foreground" { $foreground = $true; continue }
        "--no-save-state" { continue }
        "--persistent" { $persistent = $true; continue }
        "--json-command" { $jsonCommand = $true; continue }
        "--no-spawn" { $noSpawn = $true; continue }
        "--no-pretty" { $noPretty = $true; continue }
        "--sockname" {
            $i++
            $sockname = $RemainingArgs[$i]
            continue
        }
        "--statefile" {
            $i++
            $statefile = $RemainingArgs[$i]
            continue
        }
        "--logfile" {
            $i++
            $logfile = $RemainingArgs[$i]
            continue
        }
        "--server-encoding" {
            $i++
            $serverEncoding = $RemainingArgs[$i]
            continue
        }
        "--output-encoding" {
            $i++
            $outputEncoding = $RemainingArgs[$i]
            continue
        }
        default {
            Write-Stderr "chunkhound fake watchman runtime: unsupported flag $arg" 64
        }
    }
}

$clientMode = $persistent -or $jsonCommand -or $noSpawn -or $noPretty -or $serverEncoding -or $outputEncoding

if ($clientMode) {
    if ($foreground -or $statefile -or $logfile) {
        Write-Stderr "chunkhound fake watchman runtime: unsupported mixed sidecar/client flags" 64
    }
    if (-not $sockname -or -not $persistent -or -not $jsonCommand -or -not $noSpawn -or -not $noPretty -or $serverEncoding -ne "json" -or $outputEncoding -ne "json") {
        Write-Stderr "chunkhound fake watchman runtime: missing client session flags" 64
    }
    if (-not (Test-Path $sockname)) {
        Write-Stderr "chunkhound fake watchman runtime: private socket is missing" 69
    }

    $missingCapability = $env:CHUNKHOUND_FAKE_WATCHMAN_MISSING_CAPABILITY
    $watchRootOverride = $env:CHUNKHOUND_FAKE_WATCHMAN_WATCH_ROOT
    $relativePathOverride = $env:CHUNKHOUND_FAKE_WATCHMAN_RELATIVE_PATH
    $emitPdu = $env:CHUNKHOUND_FAKE_WATCHMAN_EMIT_SUBSCRIPTION_PDU
    $emitLog = $env:CHUNKHOUND_FAKE_WATCHMAN_EMIT_LOG_AFTER_SUBSCRIBE

    while (($line = [Console]::In.ReadLine()) -ne $null) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $command = $line | ConvertFrom-Json
        } catch {
            Write-Stderr "chunkhound fake watchman runtime: invalid JSON command" 65
        }
        if (-not $command -or $command.Count -lt 1) {
            Emit-Json @{ error = "expected command array" }
            continue
        }

        $commandName = [string]$command[0]
        switch ($commandName) {
            "version" {
                $capabilities = @{
                    "cmd-watch-project" = ($missingCapability -ne "cmd-watch-project")
                    "relative_root" = ($missingCapability -ne "relative_root")
                }
                if ($command.Count -gt 1 -and $command[1].PSObject.Properties.Name -contains "required") {
                    foreach ($required in @($command[1].required)) {
                        $name = [string]$required
                        if (-not $capabilities.ContainsKey($name)) {
                            $capabilities[$name] = ($missingCapability -ne $name)
                        }
                    }
                }
                Emit-Json @{ version = $version; capabilities = $capabilities }
            }
            "watch-project" {
                if ($command.Count -lt 2) {
                    Emit-Json @{ error = "watch-project requires a target path" }
                    continue
                }
                $targetPath = [string]$command[1]
                $response = @{
                    version = $version
                    watch = $(if ($watchRootOverride) { $watchRootOverride } else { $targetPath })
                }
                if ($relativePathOverride -and $relativePathOverride -ne ".") {
                    $response.relative_path = $relativePathOverride.Replace("\", "/")
                }
                Emit-Json $response
            }
            "subscribe" {
                if ($command.Count -lt 3) {
                    Emit-Json @{ error = "subscribe requires a root and subscription name" }
                    continue
                }
                $subscriptionName = [string]$command[2]
                $watchRoot = [string]$command[1]
                Emit-Json @{ version = $version; subscribe = $subscriptionName }
                if ($emitLog) {
                    Emit-Json @{ log = $emitLog }
                }
                if ($emitPdu) {
                    try {
                        $payload = $emitPdu | ConvertFrom-Json
                    } catch {
                        $payload = @{
                            subscription = $subscriptionName
                            root = $watchRoot
                            clock = "c:0:1"
                            files = @(
                                @{
                                    name = "src/example.py"
                                    exists = $true
                                    new = $true
                                    type = "f"
                                }
                            )
                        }
                    }
                    if ($payload -isnot [System.Collections.IDictionary]) {
                        $payload = @{
                            subscription = $subscriptionName
                            root = $watchRoot
                            clock = "c:0:1"
                            files = @()
                        }
                    }
                    if (-not $payload.Contains("subscription")) {
                        $payload["subscription"] = $subscriptionName
                    }
                    if (-not $payload.Contains("root")) {
                        $payload["root"] = $watchRoot
                    }
                    Emit-Json $payload
                }
            }
            default {
                Emit-Json @{ error = "unsupported command $commandName" }
            }
        }
    }
    exit 0
}

if (-not $foreground -or -not $sockname -or -not $statefile -or -not $logfile) {
    Write-Stderr "chunkhound fake watchman runtime: missing sidecar path flags" 64
}

if ($env:CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS) {
    Start-Sleep -Seconds ([double]$env:CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS)
}

$logfileDir = Split-Path -Parent $logfile
if ($logfileDir) {
    New-Item -ItemType Directory -Force -Path $logfileDir | Out-Null
}
Add-Content -Path $logfile -Value "fake watchman start"

if ($env:CHUNKHOUND_FAKE_WATCHMAN_FAIL_BEFORE_READY -eq "1") {
    exit 70
}

$socketDir = Split-Path -Parent $sockname
if ($socketDir) {
    New-Item -ItemType Directory -Force -Path $socketDir | Out-Null
}
$statefileDir = Split-Path -Parent $statefile
if ($statefileDir) {
    New-Item -ItemType Directory -Force -Path $statefileDir | Out-Null
}
Set-Content -Path $sockname -Value "socket ready"
Set-Content -Path $statefile -Value "state ready"

while ($true) {
    Start-Sleep -Seconds 1
}
