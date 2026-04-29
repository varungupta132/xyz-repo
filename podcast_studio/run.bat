@echo off
echo.
echo  ==========================================
echo   INTERACTIVE PODCAST STUDIO
echo  ==========================================
echo   Starting server...
echo   Open: http://localhost:8080
echo  ==========================================
echo.
cd /d "%~dp0"
python app.py
pause
