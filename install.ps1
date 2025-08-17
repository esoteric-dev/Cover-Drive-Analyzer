# Cricket Cover Drive Analyzer - Setup Script
Write-Host "========================================" -ForegroundColor Green
Write-Host "Cricket Cover Drive Analyzer - Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Install dependencies
Write-Host "Installing required dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green
    } else {
        throw "Failed to install dependencies"
    }
} catch {
    Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Run system test
Write-Host "Running system test..." -ForegroundColor Yellow
try {
    python test_system.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ System test passed!" -ForegroundColor Green
    } else {
        throw "System test failed"
    }
} catch {
    Write-Host "✗ System test failed" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now use the Cricket Cover Drive Analyzer:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Single video analysis:" -ForegroundColor White
Write-Host "   python cricket_cover_drive_analyzer.py your_video.mp4" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Batch analysis:" -ForegroundColor White
Write-Host "   python batch_analyzer.py video_folder/" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Video validation:" -ForegroundColor White
Write-Host "   python video_utils.py validate your_video.mp4" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Demo mode:" -ForegroundColor White
Write-Host "   python demo.py" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit"
