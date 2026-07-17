# Activate DuckDB from a LanceDB ChunkHound index (Windows PowerShell).
#
# Forwards all args to convert_lancedb_to_duckdb.py via uv without treating the
# repo root as the project path. Relative --project/--source/--dest are resolved
# against the caller's working directory.
#
# What -Activate / --activate does:
#   Converts Lance → DuckDB, then sets .chunkhound.json to provider=duckdb.
#   Lance data is left in place.
#
# Examples:
#   .\scripts\activate_duckdb_from_lancedb.ps1 --project . --activate --overwrite
#   .\scripts\activate_duckdb_from_lancedb.ps1 --project C:\work\myrepo --activate --overwrite
#   .\scripts\activate_duckdb_from_lancedb.ps1 --help

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$CallerCwd = (Get-Location).Path

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found on PATH (required to run the converter)"
}

# Absolutize relative path flags against caller CWD
$out = New-Object System.Collections.Generic.List[string]
$i = 0
$raw = @($args)
while ($i -lt $raw.Count) {
    $a = [string]$raw[$i]
    if ($a -in @("--project", "--source", "--dest")) {
        if ($i + 1 -ge $raw.Count) {
            Write-Error "$a requires a value"
        }
        $val = [string]$raw[$i + 1]
        if (-not [System.IO.Path]::IsPathRooted($val)) {
            $val = Join-Path $CallerCwd $val
        }
        $out.Add($a) | Out-Null
        $out.Add($val) | Out-Null
        $i += 2
        continue
    }
    $out.Add($a) | Out-Null
    $i += 1
}

$py = Join-Path $ScriptDir "convert_lancedb_to_duckdb.py"
& uv run --directory $RepoRoot python $py @($out.ToArray())
exit $LASTEXITCODE
