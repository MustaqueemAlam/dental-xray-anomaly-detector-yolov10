#!/usr/bin/env bash
# exit on error
set -o errexit

# This script runs on the server when you deploy.

# 1. Install Node.js dependencies
npm install

# 2. Install Python and dependencies from requirements.txt
pip install --upgrade pip
pip install -r python_inference/requirements.txt

echo "Build finished successfully!"
