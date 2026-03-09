@echo off
if "%~1"=="--version" (
    echo watchman 0.0.0-chunkhound-placeholder
    exit /b 0
)

echo chunkhound fake watchman runtime: only --version is supported 1>&2
exit /b 64
