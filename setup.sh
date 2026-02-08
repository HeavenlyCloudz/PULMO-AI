#!/bin/bash
# Setup script for PULMO AI Microwave Scanner

echo "Setting up PULMO AI Microwave Scanner..."
echo "========================================"

# Create directory structure
mkdir -p pulmo_ai_scanner/{hardware,scanning,processing,data/{datasets,logs}}

# Install required packages
echo "Installing Python packages..."
pip3 install numpy pandas matplotlib scipy pyserial

# Set up GPIO access (if on Raspberry Pi)
if [ -f /etc/rpi-issue ]; then
    echo "Detected Raspberry Pi. Setting up GPIO..."
    pip3 install RPi.GPIO
    # Add user to GPIO group
    sudo usermod -a -G gpio $USER
fi

# Make scripts executable
chmod +x main.py quick_start.py

echo ""
echo "Setup complete!"
echo ""
echo "To run the scanner:"
echo "  cd pulmo_ai_scanner"
echo "  python3 main.py"
echo ""
echo "For a quick test:"
echo "  python3 quick_start.py"