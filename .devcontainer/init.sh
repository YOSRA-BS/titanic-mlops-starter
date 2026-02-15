#!/usr/bin/env bash
set -euo pipefail

echo "Running post-create initialization: installing Python dependencies..."

# Ensure we're in the workspace
cd /workspace || exit 1

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

if [ -f /workspace/requirements.txt ]; then
  echo "Installing from requirements.txt..."
  pip install -r /workspace/requirements.txt
else
  echo "No requirements.txt found at /workspace/requirements.txt — skipping."
fi

if [ -f /workspace/setup.py ] || [ -f /workspace/pyproject.toml ]; then
  echo "Installing package in editable mode..."
  pip install -e /workspace
else
  echo "No setup.py or pyproject.toml found — skipping 'pip install -e .'."
fi

echo "Initialization complete."
