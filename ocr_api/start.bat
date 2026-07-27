@echo off
REM ──────────────────────────────────────────────────────
REM  Start the OCR Output API using the AI-OCR venv
REM ──────────────────────────────────────────────────────

set VENV_PYTHON=C:\Users\anubhavmznac\Desktop\AI-OCR\venv\Scripts\python.exe

echo.
echo ============================================
echo   OCR Output API  —  Chandra OCR-2
echo ============================================
echo.

if not exist "%VENV_PYTHON%" (
    echo ERROR: Python not found at %VENV_PYTHON%
    echo Please check the AI-OCR virtual environment.
    pause
    exit /b 1
)

echo Using Python: %VENV_PYTHON%
echo Starting server on http://0.0.0.0:8000
echo Swagger docs: http://127.0.0.1:8000/docs
echo.

cd /d %~dp0\..
"%VENV_PYTHON%" -m uvicorn ocr_api.main:app --host 0.0.0.0 --port 8000 --reload

pause
