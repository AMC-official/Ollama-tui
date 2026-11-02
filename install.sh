#!/bin/bash

# Exit on error
set -e

# Check if we are in the source directory or the release archive
if [ -f "main.go" ]; then
    echo "Source code detected. Building ollama-tui..."
    go build -o ollama-tui main.go
elif [ -f "ollama-tui" ]; then
    echo "Pre-compiled binary detected."
else
    echo "Error: Neither main.go nor ollama-tui found in this directory."
    exit 1
fi

echo "Installing to /usr/local/bin (requires sudo)..."
sudo install -m 755 ollama-tui /usr/local/bin/ollama-tui

echo "Done! You can now run 'ollama-tui' from anywhere."
