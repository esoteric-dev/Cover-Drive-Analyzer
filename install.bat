@echo off
echo ========================================
echo Cricket Cover Drive Analyzer - Setup
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

echo Installing required dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Running system test...
python test_system.py
if %errorlevel% neq 0 (
    echo System test failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo You can now use the Cricket Cover Drive Analyzer:
echo.
echo 1. Single video analysis:
echo    python cricket_cover_drive_analyzer.py your_video.mp4
echo.
echo 2. Batch analysis:
echo    python batch_analyzer.py video_folder/
echo.
echo 3. Video validation:
echo    python video_utils.py validate your_video.mp4
echo.
echo 4. Demo mode:
echo    python demo.py
echo.
pause
