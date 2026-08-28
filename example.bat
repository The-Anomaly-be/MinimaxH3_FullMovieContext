@echo off
REM Example launch for H3 T2V 15s (3 clips of 5s chained via Motion Context).
REM Prerequisite: ComfyUI must be running with the required custom nodes + models (see README.md).
cd /d "%~dp0"
python render.py examples\prompt.txt --host http://127.0.0.1:8192 --out output
pause
