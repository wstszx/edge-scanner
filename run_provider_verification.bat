@echo off
setlocal
cd /d "%~dp0"
python provider_verification.py --summary-only %*
exit /b %ERRORLEVEL%
