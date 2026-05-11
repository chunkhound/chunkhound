@echo off
setlocal ENABLEEXTENSIONS
where python >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
    python -m chunkhound.watchman_runtime.bridge %*
    set EXITCODE=%ERRORLEVEL%
    exit /B %EXITCODE%
)
py -3 -m chunkhound.watchman_runtime.bridge %*
set EXITCODE=%ERRORLEVEL%
exit /B %EXITCODE%
