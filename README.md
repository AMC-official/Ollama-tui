# Ollama-TUI 🦙💻

A blazing-fast, zero-bloat terminal user interface (TUI) for [Ollama](https://ollama.com/), built entirely in Go.

Born out of a desire to ditch heavy Electron apps and web browsers, **Ollama-TUI** provides a sleek, Vim-friendly, fully native chat experience right inside your terminal. Built and tested primarily on FreeBSD, it aims to be fast, lightweight, and keyboard-driven. and to be as crossplatform as possible, running on Windows, MacOS, Linux, and BSD systems

> Note: DeepSeek was partially used in the process of making this project to assist with development and iteration.

## ✨ Features

* **Native & Fast:** Built with [Bubble Tea](https://github.com/charmbracelet/bubbletea) and [Lipgloss](https://github.com/charmbracelet/lipgloss). No web wrappers, no Electron, just pure terminal speed.
* **Auto-Server Management:** Automatically boots `ollama serve` in the background when you launch it. Prompts to kill the server when you exit.
* **Multi-Model Support:** Switch between your installed local models on the fly without losing your chat context.
* **System Prompts:** Set a persistent system prompt (e.g., *"You are a UNIX graybeard"*) using the built-in editor.
* **Context Meter:** Keep track of your token usage with a visual context window meter in the header.
* **Chat History:** Save your chats to `.json` files and load them back up anytime.
* **Branching:** Go back in time, edit a past message, and branch the conversation like ChatGPT.
* **Vim-Style Navigation:** Use `Tab` to toggle between typing and "List Mode" to scroll and select messages.
* **Clipboard Integration:** Copy selected AI responses directly to your system clipboard.
* **Cancelable Streams:** Stop the AI mid-generation if it starts yapping too much.
* **Text Attachment Support:** Attach files by entering their path, including text-based PDFs (not scanned images), `.txt`, `.docx`, `.odt`, and `.abw` documents and script files such as `.py` or `.conf`. You can add, remove, or edit the attachment list before sending a message.

## 📦 Prerequisites

Before you build or run this, ensure you have:
1. **Go** (1.20 or newer) installed.
2. **Ollama** installed and available in your `$PATH`.
3. **At least one model pulled** (e.g., `ollama pull gemma3:1b`).

### 🐧 POSIX (FreeBSD, Linux, macOS) Dependencies

To copy text to the clipboard, the `atotto/clipboard` library requires `xclip`.

* **FreeBSD:** `sudo pkg install xclip`
* **Debian/Ubuntu:** `sudo apt install xclip`
* **RHEL/Fedora:** `sudo dnf install xclip`
* **Arch/Artix:** `sudo pacman -S xclip`
* **macOS:** No extra dependencies needed.

---

## 🚀 Installation

### 1. Clone & Build

```bash
git clone https://github.com/AMC-official/Ollama-tui.git
cd Ollama-tui
go mod tidy
```

### 2. Run the Installer

For POSIX (Linux, FreeBSD, macOS):

```bash
chmod +x install.sh
./install.sh
```

For Windows (PowerShell):

```powershell
.\install.ps1
```

### 3. Manual Run (Without Installing)

```bash
go run main.go
```

---

## ⌨️ Keybindings

Ollama-TUI uses a modal interface. You are either in Chat Mode (typing) or List Mode (navigating).

| Key | Mode | Action |
| --- | --- | --- |
| Tab / Esc | Any | Toggle between Chat Mode and List Mode |
| Ctrl + F | Chat | Upload attachments by writing a path or drag-and-dropping files |
| Ctrl + L | Chat | List attached attachments and edit them |
| Enter | Chat | Send message to AI |
| Alt + S | Any | Cancel the AI mid-generation |
| w / z / s | List | Scroll up / down (Supports AZERTY & QWERTY) |
| Up / Down | List | Select a specific message |
| Shift + C | List | Copy the selected message to clipboard |
| Shift + N | List | Start a completely new chat |
| Shift + S | List | Switch to a different Ollama model |
| Shift + P | List | Open the System Prompt editor |
| Shift + J | List | Save current chat to `.json` |
| Shift + B | List | Branch the chat from the selected message |
| Ctrl + O | List | Open the Chat History browser |
| Ctrl + C | Any | Exit app (prompts to kill Ollama server) |

## 🖥️ Platform Support

* FreeBSD / OpenBSD / NetBSD: Fully supported natively.
* Linux (Arch, Debian, Fedora, etc.): Fully supported natively.
* macOS (Intel & Apple Silicon): Fully supported natively.
* Windows: Fully supported natively via cross-compilation (requires the `.exe` build).

> Note: The Windows installer script is separate because Windows is not POSIX-compliant and handles background processes slightly differently.

## 🛠️ Built With

* Bubble Tea - The TUI framework.
* Lipgloss - Style definitions for nice terminal layouts.
* Ollama API - The local AI backend.

> Note: In v1.1.3 and earlier, the `install.ps1` file may appear in non-Windows archives due to a `goreleaser.yml` misconfiguration. This is harmless and will be fixed in future releases.

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.
