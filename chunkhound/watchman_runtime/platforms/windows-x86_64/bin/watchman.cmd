@echo off
setlocal EnableExtensions EnableDelayedExpansion
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0watchman.ps1" %*
exit /b %ERRORLEVEL%
