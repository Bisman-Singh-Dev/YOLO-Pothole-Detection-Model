@echo off
title YOLO Pothole Detection & Safety System - Web Dashboard
echo =================================================================
echo   🛣️ Launching YOLO Pothole Detection Web Dashboard...
echo =================================================================
echo.
echo Starting Streamlit web application on http://localhost:8501 ...
echo.
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Attempting direct Python runner...
    python app.py
)
pause
