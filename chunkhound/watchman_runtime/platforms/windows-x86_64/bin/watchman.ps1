param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $python) {
    & $python.Source -m chunkhound.watchman_runtime.bridge @RemainingArgs
    exit $LASTEXITCODE
}

& py -3 -m chunkhound.watchman_runtime.bridge @RemainingArgs
exit $LASTEXITCODE
