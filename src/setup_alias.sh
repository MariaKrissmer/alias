#!/bin/bash

# Specify the directory for the virtual environment
VENV_DIR="./alias"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip

# Check if requirements.txt exists before installing
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping package installation."
fi

echo "Setup is complete. Virtual environment is ready and packages are installed."
