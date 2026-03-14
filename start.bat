@echo off
echo ============================================
echo   STRATIFY - Starting All Servers
echo ============================================
echo.

REM Change to STRATIFY directory
cd /d "%~dp0"

REM 1. Start FastAPI Bridge (port 8000)
echo [1/3] Starting FastAPI Bridge on port 9000...
start "Stratify API" cmd /k "uvicorn backend.api:app --reload --port 9000"
timeout /t 2 /nobreak >nul

echo [2/3] Streamlit Dashboard (Background Engine Ready)
REM start "Stratify Streamlit" cmd /k "streamlit run app.py --server.port 8501"
timeout /t 1 /nobreak >nul

REM 3. Start React Frontend (port 5173)
echo [3/3] Starting React Frontend on port 5173...
start "Stratify Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================
echo   All PREMIUM servers starting! (Option B)
echo.
echo   Main App UI    : http://localhost:5173
echo   Docs / API     : http://localhost:9000/docs
echo ============================================
echo.
echo Close this window or press any key when done.
pause >nul
