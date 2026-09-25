#!/bin/bash
# 01_setup_and_download.sh

echo "Setting up Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ -f "requirements_llm.txt" ]; then
    pip install -r requirements_llm.txt
fi

echo "Downloading models..."
if [ -f "download_models.py" ]; then
    python download_models.py
fi
if [ -f "download_wget.py" ]; then
    python download_wget.py
fi

echo "Setup Complete!"
