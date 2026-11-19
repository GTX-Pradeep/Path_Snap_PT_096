@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:8080 and open it.
cd /d "%~dp0ui"
start "" http://localhost:8080
python -m http.server 8080
