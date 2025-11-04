#!/bin/bash
# Setup script for Traffic Video Analyzer

set -e

echo "Traffic Video Analyzer - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version detected"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for FFmpeg
echo ""
echo "Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -n1)
    echo "✓ FFmpeg found: $ffmpeg_version"
else
    echo "⚠ FFmpeg not found. Install it for streaming support:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
fi

# Check for CUDA (optional)
echo ""
echo "Checking for CUDA (optional)..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$cuda_available" = "True" ]; then
        echo "✓ CUDA is available for GPU acceleration"
    else
        echo "ℹ CUDA not available (CPU mode only)"
    fi
else
    echo "ℹ PyTorch not yet installed or CUDA check failed"
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python vehicle_detection.py"
echo ""
echo "For MQTT integration:"
echo "  python mqtt.py"
echo ""
