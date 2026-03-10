@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="--version" (
    echo watchman 0.0.0-chunkhound-sidecar-placeholder
    exit /b 0
)

set "sockname="
set "statefile="
set "logfile="

:parse_args
if "%~1"=="" goto parsed_args
if "%~1"=="--foreground" (
    shift
    goto parse_args
)
if "%~1"=="--no-save-state" (
    shift
    goto parse_args
)
if "%~1"=="--sockname" (
    set "sockname=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--statefile" (
    set "statefile=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--logfile" (
    set "logfile=%~2"
    shift
    shift
    goto parse_args
)
echo chunkhound fake watchman runtime: unsupported flag %~1 1>&2
exit /b 64

:parsed_args
if not defined sockname (
    echo chunkhound fake watchman runtime: missing --sockname 1>&2
    exit /b 64
)
if not defined statefile (
    echo chunkhound fake watchman runtime: missing --statefile 1>&2
    exit /b 64
)
if not defined logfile (
    echo chunkhound fake watchman runtime: missing --logfile 1>&2
    exit /b 64
)

if defined CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS (
    powershell -NoProfile -Command "Start-Sleep -Seconds $env:CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS" >nul
)

for %%F in ("%logfile%") do if not exist "%%~dpF" mkdir "%%~dpF"
>>"%logfile%" echo fake watchman start

if "%CHUNKHOUND_FAKE_WATCHMAN_FAIL_BEFORE_READY%"=="1" exit /b 70

for %%F in ("%sockname%") do if not exist "%%~dpF" mkdir "%%~dpF"
for %%F in ("%statefile%") do if not exist "%%~dpF" mkdir "%%~dpF"
>"%sockname%" echo socket ready
>"%statefile%" echo state ready

:wait_loop
timeout /t 1 /nobreak >nul
goto wait_loop
