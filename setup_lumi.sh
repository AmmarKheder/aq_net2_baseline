#!/bin/bash
# Setup script for LUMI supercomputer

echo "Setting up environment for LUMI..."

# Load required modules
module load LUMI/24.03
module load partition/G  
module load pytorch/2.4.0
module load Python/3.10.8-GCCcore-12.2.0

# Create virtual environment
python -m venv venv_lumi
source venv_lumi/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_lumi.txt

echo "Setup complete!"
echo "To activate the environment in future sessions:"
echo "1. module load LUMI/24.03 partition/G pytorch/2.4.0"
echo "2. source venv_lumi/bin/activate"
