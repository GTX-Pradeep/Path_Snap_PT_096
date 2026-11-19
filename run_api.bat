@echo off
REM Laptop 1: start the PathSnap API on port 8000 (reachable from Laptop 2).
REM Evaluation trips are read from data\evaluator_test unless PATHSNAP_EVAL_DIR is set.
cd /d "%~dp0api"
if "%PATHSNAP_EVAL_DIR%"=="" set "PATHSNAP_EVAL_DIR=%~dp0data\evaluator_test"
echo Evaluation trips: %PATHSNAP_EVAL_DIR%
python -m uvicorn main:app --host 0.0.0.0 --port 8000
