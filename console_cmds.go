package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"
)

// --- More console commands ---
//
// console.go keeps the console itself (input, history, the older commands);
// the commands added later live here, one small function each. runConsole in
// console.go only has to call them.

// consoleColors turns the yellow/green token colours on or off.
func (m *model) consoleColors(arg string) {
	switch strings.ToLower(strings.TrimSpace(arg)) {
	case "off", "no", "plain", "0":
		m.plainTokens = true
	case "on", "yes", "1":
		m.plainTokens = false
	case "":
		m.plainTokens = !m.plainTokens
	default:
		m.consolePrint("usage: :colors [on|off]")
		return
	}
	if m.plainTokens {
		m.consolePrint("✓ Token colours are off.")
		return
	}
	m.consolePrint("✓ Token colours are on: every token gets the next colour.",
		"  "+paintText("yellow and green take turns, one colour for each token", false))
}

// consoleCopy copies the last AI reply to the clipboard.
func (m *model) consoleCopy() {
	for i := len(m.messages) - 1; i >= 0; i-- {
		msg := m.messages[i]
		if msg.Role != "AI" || strings.TrimSpace(msg.Content) == "" {
			continue
		}
		text := strings.TrimSuffix(msg.Content, " [Stopped by user]")
		if err := clipboard.WriteAll(text); err != nil {
			m.consolePrint("✗ Copy failed: " + err.Error() + " (is xclip installed?)")
			return
		}
		m.consolePrint(fmt.Sprintf("✓ Copied the last AI reply (%d characters).", len([]rune(text))))
		return
	}
	m.consolePrint("✗ There is no AI reply to copy yet.")
}

// --- :export ---

// exportDir holds the chats exported as Markdown.
func exportDir() string { return filepath.Join(dataDir(), "exports") }

// chatMarkdown writes a chat as a Markdown document.
func chatMarkdown(msgs []ChatMessage, modelName, backend string) string {
	var b strings.Builder
	b.WriteString("# Ollama-TUI chat\n\n")
	b.WriteString(fmt.Sprintf("Model: %s (%s) • exported %s\n\n", modelName, backend, time.Now().Format("2006-01-02 15:04")))
	for _, msg := range msgs {
		switch msg.Role {
		case "You":
			b.WriteString("## You\n\n" + msg.Content + "\n\n")
			if msg.Context != "" {
				b.WriteString("Attached text:\n\n~~~~text\n" + strings.TrimRight(msg.Context, "\n") + "\n~~~~\n\n")
			}
		case "AI":
			b.WriteString("## AI\n\n")
			if strings.TrimSpace(msg.Thinking) != "" {
				b.WriteString("<details><summary>Thinking</summary>\n\n" + msg.Thinking + "\n\n</details>\n\n")
			}
			if strings.TrimSpace(msg.Content) != "" {
				b.WriteString(msg.Content + "\n\n")
			}
			if len(msg.ToolCalls) > 0 {
				b.WriteString("*Tool calls: " + describeCalls(msg.ToolCalls) + "*\n\n")
			}
			if msg.Elapsed != "" {
				b.WriteString("*Reply took " + msg.Elapsed + "*\n\n")
			}
		case "Tool":
			b.WriteString("### Tool result: " + msg.ToolName + "\n\n~~~~text\n" + strings.TrimRight(msg.Content, "\n") + "\n~~~~\n\n")
		default:
			b.WriteString("> " + strings.ReplaceAll(msg.Content, "\n", "\n> ") + "\n\n")
		}
	}
	return b.String()
}

// consoleExport saves the chat as a Markdown file.
func (m *model) consoleExport(arg string) {
	if len(m.messages) == 0 {
		m.consolePrint("✗ The chat is empty, nothing to export.")
		return
	}
	dir := exportDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		m.consolePrint("✗ " + err.Error())
		return
	}
	file := sanitizeFileName(arg)
	if file == "" {
		file = "chat_" + time.Now().Format("20060102_150405")
	}
	if !strings.HasSuffix(file, ".md") {
		file += ".md"
	}
	path := filepath.Join(dir, file)
	text := chatMarkdown(m.messages, m.currentModelName(), backendLabel(m.backend))
	if err := os.WriteFile(path, []byte(text), 0600); err != nil {
		m.consolePrint("✗ " + err.Error())
		return
	}
	m.consolePrint("✓ Exported the chat to " + path)
}

// --- :set and :system ---

// settingsLines shows the sampling settings and the system prompt.
func (m model) settingsLines() []string {
	orDefault := func(s string) string {
		if s == "" {
			return "(model default)"
		}
		return s
	}
	topK, ctx := "", ""
	if m.config.TopK != nil {
		topK = strconv.Itoa(*m.config.TopK)
	}
	if m.config.NumCtx > 0 {
		ctx = strconv.Itoa(m.config.NumCtx)
	}
	sys := "(none)"
	if strings.TrimSpace(m.systemPrompt) != "" {
		sys = shorten(m.systemPrompt, 80)
	}
	return []string{
		"Temperature:    " + orDefault(formatFloat(m.config.Temperature)),
		"Top P:          " + orDefault(formatFloat(m.config.TopP)),
		"Top K:          " + orDefault(topK),
		"Context tokens: " + orDefault(ctx),
		"System prompt:  " + sys,
		"(:set <temp|top_p|top_k|ctx> <value|default> changes one, :system <text|clear> the prompt)",
	}
}

// consoleSet changes one sampling setting and saves it, with the same limits
// as the Settings screen.
func (m *model) consoleSet(arg string) {
	fields := strings.Fields(arg)
	if len(fields) == 0 {
		m.consolePrint(m.settingsLines()...)
		return
	}
	if len(fields) < 2 {
		m.consolePrint("usage: :set <temp|top_p|top_k|ctx> <value|default>")
		return
	}
	key, val := strings.ToLower(fields[0]), strings.ToLower(fields[1])
	reset := val == "default" || val == "none" || val == "off" || val == "-"
	cfg := m.config

	switch key {
	case "temp", "temperature":
		if reset {
			cfg.Temperature = nil
			break
		}
		v, err := strconv.ParseFloat(val, 64)
		if err != nil || v < 0 || v > 2 {
			m.consolePrint("✗ temperature must be a number from 0 to 2")
			return
		}
		cfg.Temperature = &v

	case "top_p", "topp", "top-p":
		if reset {
			cfg.TopP = nil
			break
		}
		v, err := strconv.ParseFloat(val, 64)
		if err != nil || v < 0 || v > 1 {
			m.consolePrint("✗ top P must be a number from 0 to 1")
			return
		}
		cfg.TopP = &v

	case "top_k", "topk", "top-k":
		if reset {
			cfg.TopK = nil
			break
		}
		v, err := strconv.Atoi(val)
		if err != nil || v < 1 {
			m.consolePrint("✗ top K must be a whole number, 1 or more")
			return
		}
		cfg.TopK = &v

	case "ctx", "context", "num_ctx":
		if reset {
			cfg.NumCtx = 0
			break
		}
		v, err := strconv.Atoi(val)
		if err != nil || v < 512 {
			m.consolePrint("✗ context tokens must be a whole number, 512 or more")
			return
		}
		if m.modelMaxCtx > 0 && v > m.modelMaxCtx {
			m.consolePrint(fmt.Sprintf("✗ this model supports at most %d context tokens", m.modelMaxCtx))
			return
		}
		cfg.NumCtx = v

	default:
		m.consolePrint(fmt.Sprintf("✗ Unknown setting %q. Use temp, top_p, top_k or ctx.", key))
		return
	}

	m.config = cfg
	saveConfig(cfg)
	lines := append([]string{"✓ Saved."}, m.settingsLines()...)
	if m.isCloud() && (key == "ctx" || key == "context" || key == "num_ctx") {
		lines = append(lines, "(cloud backends ignore the context setting: they use the model's own limit)")
	}
	m.consolePrint(lines...)
}

// consoleSystem shows, sets or clears the system prompt.
func (m *model) consoleSystem(arg string) {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		if strings.TrimSpace(m.systemPrompt) == "" {
			m.consolePrint("No system prompt is set. :system <text> sets one (long ones fit better in the Settings screen).")
			return
		}
		m.consolePrint("System prompt:")
		for _, l := range strings.Split(m.systemPrompt, "\n") {
			m.consolePrint("  " + l)
		}
		return
	}
	if strings.EqualFold(arg, "clear") {
		arg = ""
	}
	m.systemPrompt = arg
	m.config.SystemPrompt = arg
	saveConfig(m.config)
	if arg == "" {
		m.consolePrint("✓ System prompt cleared.")
		return
	}
	m.consolePrint("✓ System prompt saved and used from the next message on.")
}

// --- :images and :view ---

// consoleImages lists the pictures of the chat.
func (m *model) consoleImages() {
	imgs := m.chatImages()
	if len(imgs) == 0 {
		m.consolePrint("No images in this chat. Attach one with Ctrl+F, or use :view <path to a picture>.")
		return
	}
	lines := []string{"Images in this chat (:view <number> shows one):"}
	for i, ref := range imgs {
		where := ref.Path
		if where == "" {
			where = "(path unknown)"
		}
		lines = append(lines, fmt.Sprintf("  %d. %s   %s", i+1, ref.Name, where))
	}
	lines = append(lines, "Pictures are shown with: "+detectImageProtocol(os.Getenv).String())
	m.consolePrint(lines...)
}

// consoleViewImage shows a picture: by number (see :images), by path, or the
// newest one of the chat when no argument is given.
func (m *model) consoleViewImage(arg string) tea.Cmd {
	arg = strings.TrimSpace(arg)
	imgs := m.chatImages()
	var ref imageRef
	switch {
	case arg == "":
		if len(imgs) == 0 {
			m.consolePrint("✗ No images in this chat. Use :view <path to a picture>.")
			return nil
		}
		ref = imgs[len(imgs)-1]
	case isImagePath(cleanImagePath(arg)):
		p := cleanImagePath(arg)
		ref = imageRef{Name: filepath.Base(p), Path: p}
	default:
		n, err := strconv.Atoi(arg)
		if err != nil || n < 1 || n > len(imgs) {
			m.consolePrint(fmt.Sprintf("✗ Pick a number from 1 to %d (see :images) or give the path of a picture.", len(imgs)))
			return nil
		}
		ref = imgs[n-1]
	}
	m.consolePrint("Opening " + ref.Name + "...")
	return m.viewImage(ref)
}
