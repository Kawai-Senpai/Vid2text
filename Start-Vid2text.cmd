@echo off
cd /d "%~dp0"
uv run --extra cuda python app.py
if errorlevel 1 pause
