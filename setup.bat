@echo off
REM Setup script for Traffic Video Analyzer (Windows)

echo Traffic Video Analyzer - Setup Script
echo ======================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Check for FFmpeg
echo.
echo Checking for FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found. Install it for streaming support:
    echo Download from https://ffmpeg.org/download.html
) else (
    echo FFmpeg found
    ffmpeg -version | findstr /C:"version"
)

echo.
echo ======================================
echo Setup complete!
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To run the application:
echo   python vehicle_detection.py
echo.
pause
