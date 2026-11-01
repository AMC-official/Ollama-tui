#!/bin/sh

# Exit on error
set -e

echo "Building ollama-tui..."
go build -o ollama-tui main.go

echo "Installing to /usr/local/bin (requires sudo)..."
sudo mv ollama-tui /usr/local/bin/

echo "Done! You can now run 'ollama-tui' from anywhere."
