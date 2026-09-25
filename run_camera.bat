@echo off
title YOLO Pothole Detection - Live Camera Tester
echo =================================================================
echo   📹 YOLO Pothole Detection - Live Camera Testing Mode
echo =================================================================
echo.
echo Launching live camera feed using weights/best.pt...
echo.
echo Controls in the camera window:
echo   [Q] or [ESC] - Quit / Close camera
echo   [S]         - Save snapshot to runs/detect/camera_snapshots/
echo   [+] or [=]   - Increase confidence threshold
echo   [-] or [_]   - Decrease confidence threshold
echo   [H]         - Toggle HUD overlay
echo.
python test_camera.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️ An error occurred while accessing the camera.
    echo Please make sure your camera is plugged in and accessible.
)
pause
