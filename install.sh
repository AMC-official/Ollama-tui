#!/bin/sh
# Installs ollama-tui.
#
# Works from a source checkout (builds every .go file in this folder) and from
# a release archive (installs the ready-made binary). Plain POSIX sh on
# purpose, so it runs on Linux, the BSDs and macOS.
#
#   ./install.sh                         installs to /usr/local/bin
#   PREFIX=$HOME/.local ./install.sh     installs to ~/.local/bin (no sudo needed)

set -eu

NAME="ollama-tui"
PREFIX="${PREFIX:-/usr/local}"
BINDIR="$PREFIX/bin"

# Work from the folder this script lives in, wherever it was started from.
cd "$(dirname "$0")"

have() { command -v "$1" >/dev/null 2>&1; }

# 1. Get a binary: build it from source, or use the one from the release archive.
if [ -f go.mod ] && have go; then
    echo "Source code detected. Building $NAME (all the .go files in this folder)..."
    # -s -w drops debug info and -trimpath drops build paths: a smaller binary.
    go build -trimpath -ldflags "-s -w" -o "$NAME" .
elif [ -f "$NAME" ]; then
    echo "Pre-compiled binary detected."
elif [ -f go.mod ]; then
    echo "Error: this is the source code, but Go is not installed (see https://go.dev/dl/)." >&2
    exit 1
else
    echo "Error: found neither the source code (go.mod) nor a pre-compiled '$NAME' in this folder." >&2
    exit 1
fi

# 2. Only ask for more privileges when the target folder needs them.
SUDO=""
mkdir -p "$BINDIR" 2>/dev/null || true
if [ ! -w "$BINDIR" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        if have sudo; then
            SUDO="sudo"
        elif have doas; then
            SUDO="doas"
        else
            echo "Error: cannot write to $BINDIR, and neither sudo nor doas was found." >&2
            echo "Run this script as root, or pick a folder you own: PREFIX=\$HOME/.local $0" >&2
            exit 1
        fi
    fi
    echo "Installing to $BINDIR (requires ${SUDO:-root})..."
else
    echo "Installing to $BINDIR..."
fi

$SUDO mkdir -p "$BINDIR"
$SUDO install -m 755 "$NAME" "$BINDIR/$NAME"

echo "Done! You can now run '$NAME' from anywhere."

# 3. Friendly notes.
case ":$PATH:" in
    *":$BINDIR:"*) ;;
    *) echo "Note: $BINDIR is not in your PATH yet." ;;
esac
if [ "$(uname -s)" != "Darwin" ] && ! have xclip && ! have xsel; then
    echo "Tip: install xclip (or xsel) so Shift+C can copy to the clipboard."
fi
