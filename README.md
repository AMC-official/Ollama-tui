# Ollama-TUI 🦙💻

<p align="center">
  <img src="https://github.com/AMC-official/Ollama-tui/raw/main/assets/ollama-tui-logo.svg" alt="Ollama-TUI ASCII Icon" width="180" />
</p>

<p align="center">
<i>The official logo of Ollama-TUI.</i>
</p>

## 📸 Screenshots

### Startup

<p align="center">
  <img src="assets/ollama-tui-startup.png" alt="Startup Screen" width="900">
</p>

<p align="center">
<i>Startup screen with model selection and automatic server management.</i>
</p>

---

### Chat Mode

<p align="center">
  <img src="assets/ollama-tui-usage-chat-mode.png" alt="Chat Mode" width="900">
</p>

<p align="center">
<i>Native markdown rendering, streaming responses, context meter, and attachments.</i>
</p>

---

### List Mode

<p align="center">
  <img src="assets/ollama-list-mode.png" alt="List Mode" width="900">
</p>

<p align="center">
<i>Browse conversations, copy responses, branch chats, and navigate with Vim-style controls.</i>
</p>

---

### Settings

<p align="center">
  <img src="assets/ollama-inter-settings.png" alt="Settings Screen" width="900">
</p>

<p align="center">
<i>Configure the system prompt, temperature, top-P, top-K, and context token limit from one place.</i>
</p>

--------------------------------------------

Ollama-TUI is a fast, keyboard-first terminal client for [Ollama](https://ollama.com/) built entirely in Go. It focuses on responsiveness, native terminal rendering, and a workflow that feels at home in a shell rather than a browser.

Born out of a desire to ditch heavy Electron apps and web browsers, **Ollama-TUI** provides a sleek, Vim-friendly, fully native chat experience right inside your terminal. Built and tested primarily on Linux and FreeBSD for maximum speed and reliability.

> Note: DeepSeek was partially used in the process of making this project to assist with development and iteration.

## ✨ Features

* **Native & Fast:** Built with [Bubble Tea](https://github.com/charmbracelet/bubbletea) and [Lipgloss](https://github.com/charmbracelet/lipgloss). No web wrappers, no Electron, just pure terminal speed.
* **Auto-Server Management:** Automatically boots `ollama serve` in the background when you launch it. Prompts to kill the server when you exit.
* **Multi-Model Support:** Switch between your installed local models on the fly without losing your chat context.
* **System Prompts:** Set a persistent system prompt (e.g., *"You are a UNIX graybeard"*) using the built-in editor.
* **Context Meter:** Displays current context usage based on your configured context limit.
* **Chat History:** Save your chats to `.json` files and load them back up anytime.
* **Persistent Conversation Context:** Uses Ollama's chat API so the model remembers the full conversation.
* **Branching:** Go back in time, edit a past message, and branch the conversation like ChatGPT.
* **Vim-Style Navigation:** Use `Tab` to toggle between typing and "List Mode" to scroll and select messages.
* **Clipboard Integration:** Copy selected AI responses directly to your system clipboard.
* **Cancelable Streams:** Stop the AI mid-generation if it starts yapping too much.
* **Text Attachment Support:** Attach files by entering their path, including text-based PDFs (not scanned images), `.txt`, `.docx`, `.odt`, and `.abw` documents and script files such as `.py` or `.sh`.
* **Capability Detection:** Automatically detects if your model supports Vision, Thinking, or Tools.
* **Thinking Viewer:** Press `Alt+T` to see the model's internal chain-of-thought (gray italic text) before the final response.
* **Toggles:** Turn Thinking (`Ctrl+T`) or Tool use (`Ctrl+Y`) on and off on the fly.
* **Vision Support:** Attach images (`.png`, `.jpg`, `.jpeg`, `.webp`) alongside text files.
* **Remote Ollama Support:** Connect to remote servers from the command line with the `host` argument or directly from the model selection screen with `Ctrl + H`.
* **Inference Settings:** Configure system prompt, temperature, top P, top K and context tokens from a built-in settings screen (`P` in List Mode).


## 📦 Prerequisites

Before you build or run this, ensure you have:
1. **Go** (1.26+ is recommended) installed.
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

### 4. Build it by yourself
```bash
git clone https://github.com/AMC-official/Ollama-tui.git
cd ~/ollama-tui
go mod tidy
go build -o ollama-tui main.go
```

### 5. Connecting to a Remote Server

You can run Ollama-TUI on one machine (like your phone or a laptop) and connect to an Ollama server running on another machine (like your main PC). 

**On the server machine (the one running Ollama):**
You must expose Ollama to your local network. By default, Ollama only listens to `localhost`. Start it with:
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```
On the client machine (the one running Ollama-TUI):
Find the server's **local IP address** (e.g., 192.168.1.50), then run:
```bash
ollama-tui --host http://192.168.1.50:11434
```

> (Alternatively, run ollama-tui and press Ctrl+H on the model selection screen to enter the IP address.)
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
| Shift + P | List | Open the Settings screen (System Prompt, Temperature, Top-P, Top-K, Context) |
| Shift + J | List | Save current chat to `.json` |
| Shift + B | List | Branch the chat from the selected message |
| Ctrl + O | List | Open the Chat History browser |
| Ctrl + C | Any | Exit app (prompts to kill Ollama server) |
| Alt + T | Chat | Toggle the display of the model's thinking process |
| Ctrl + T | Any | Toggle Thinking mode (if supported by model) |
| Ctrl + Y | Any | Toggle Tooling mode (if supported by model) |
| Ctrl + H | Model Selection | Change the Ollama host URL |

## 🖥️ Platform Support

- Tested natively and working:
  - Linux
  - FreeBSD
  - Android (tested via Termux)
- Expected to work (untested):
  - macOS / Darwin
  - Windows
- Experimental / untested:
  - NetBSD
  - OpenBSD
 
> The project is primarily developed and validated on Linux and FreeBSD. Other Unix-like platforms and Windows are expected to be compatible, but native support is not yet officially confirmed.

## Known Limitations

- Tool calling is still experimental.
- Context meter currently assumes the configured context window.
- Some inference settings are experimental.



## 🛠️ Built With

* Bubble Tea - The TUI framework.
* Lipgloss - Style definitions for nice terminal layouts.
* Glamour - The text formatting stylizer
* Ollama API - The local AI backend.

> Note: In v1.1.3 and earlier, the `install.ps1` file may appear in non-Windows archives due to a `goreleaser.yml` misconfiguration. This is harmless and will be fixed in future releases. (whoops! sorry about that.)

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.
