package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// --- Console: a command line with details about models and token usage ---

const maxConsoleLines = 4000 // a conversation dump (:chat, :dump) can be long

type consoleOutMsg struct{ lines []string }
type consoleKilledMsg struct{ lines []string }

func newConsoleInput() textinput.Model {
	ti := textinput.New()
	ti.Prompt = ": "
	ti.Placeholder = "help"
	ti.CharLimit = 200
	return ti
}

// parseConsole splits ":switch gemma3:4b" into "switch" and "gemma3:4b".
// The leading colon is optional.
func parseConsole(line string) (cmd, arg string) {
	line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), ":"))
	if line == "" {
		return "", ""
	}
	parts := strings.SplitN(line, " ", 2)
	cmd = strings.ToLower(parts[0])
	if len(parts) == 2 {
		arg = strings.TrimSpace(parts[1])
	}
	return cmd, arg
}

// resolveModel finds a model by its exact name, or by a part of the name as
// long as only one model contains it.
func resolveModel(name string, models []string) (int, error) {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "" {
		return -1, fmt.Errorf("usage: :switch <model name>")
	}
	for i, candidate := range models {
		if strings.ToLower(candidate) == name {
			return i, nil
		}
	}
	var hits []int
	for i, candidate := range models {
		if strings.Contains(strings.ToLower(candidate), name) {
			hits = append(hits, i)
		}
	}
	switch len(hits) {
	case 0:
		return -1, fmt.Errorf("no model matches %q", name)
	case 1:
		return hits[0], nil
	}
	shown := make([]string, 0, 6)
	for _, i := range hits {
		if len(shown) == 5 {
			shown = append(shown, "...")
			break
		}
		shown = append(shown, models[i])
	}
	return -1, fmt.Errorf("%q matches several models: %s", name, strings.Join(shown, ", "))
}

// capLines keeps only the last max lines (and lets go of the old ones).
func capLines(lines []string, max int) []string {
	if len(lines) > max {
		return append([]string(nil), lines[len(lines)-max:]...)
	}
	return lines
}

func consoleHelp() []string {
	return []string{
		"Commands (the leading : is optional):",
		"  :help             this list",
		"  :status           backend, model, server and chat summary",
		"  :chat <prompt>    ask the current model one question: shows what was sent, the reply token by token, and the stats",
		"  :chat             (no prompt) the whole chat as the model receives it, with the tokens coloured",
		"  :dump [full]      like :chat without a prompt; :dump full shows every line of long messages",
		"  :ctx              what fills the context window: the size of every message",
		"  :colors [on|off]  yellow/green token colours",
		"  :tokens           token usage of this run and the context window",
		"  :top              CPU, RAM, swap and GPU usage right now, with bars",
		"  :images           list the pictures of this chat",
		"  :view [n|path]    show a picture (inline in kitty, Ghostty, WezTerm, iTerm2; otherwise in the default viewer)",
		"  :copy             copy the last AI reply to the clipboard",
		"  :export [name]    save the chat as a Markdown file",
		"  :set [key value]  show or change temp, top_p, top_k or ctx",
		"  :system [text]    show, set or clear the system prompt (:system clear)",
		"  :models [text]    list the models (optionally only those containing text)",
		"  :show [model]     details about a model (default: the current one)",
		"  :ps               models loaded in memory right now (Ollama)",
		"  :unload [model]   free a model from memory (Ollama; all of them by default)",
		"  :switch <name>    switch to another model (a part of the name is enough)",
		"  :start            start the local Ollama server",
		"  :stop             stop the reply that is being generated",
		"  :kill             stop the Ollama server, if ollama-tui started it",
		"  :save [name]      save this console log as a .json file",
		"  :logs             where chats and console logs are saved, and the latest files",
		"  :clear            clear this screen",
		"  :exit             close the console (Esc or Ctrl+G work too)",
	}
}

func (m *model) consolePrint(lines ...string) {
	m.consoleLines = capLines(append(m.consoleLines, lines...), maxConsoleLines)
	m.refreshConsole()
}

func (m *model) refreshConsole() {
	lines := m.consoleLines
	if m.ccChan != nil && m.ccText != "" {
		// The reply of a :chat prompt, shown while it streams in (one colour per token).
		toks := m.ccTokens
		if len(toks) == 0 {
			toks = splitTokens(m.ccText)
		}
		live, _ := paintTokens(toks, 0, m.plainTokens)
		lines = append(append([]string(nil), lines...), "AI: "+live)
	}
	styled := make([]string, len(lines))
	for i, l := range lines {
		switch {
		case strings.HasPrefix(l, "> "):
			styled[i] = lipgloss.NewStyle().Foreground(colorCyan).Render(l)
		case strings.HasPrefix(l, "✗"):
			styled[i] = lipgloss.NewStyle().Foreground(colorRed).Render(l)
		case strings.HasPrefix(l, "✓"):
			styled[i] = lipgloss.NewStyle().Foreground(colorGreen).Render(l)
		default:
			styled[i] = l
		}
	}
	m.consoleVP.SetContent(lipgloss.NewStyle().Width(m.consoleVP.Width).Render(strings.Join(styled, "\n")))
	m.consoleVP.GotoBottom()
}

func (m *model) openConsole() {
	m.console = true
	m.consoleIn.SetValue("")
	m.consoleIn.Focus()
	if len(m.consoleLines) == 0 {
		m.consolePrint("Ollama-TUI console. Type :help for the commands, Esc goes back.", "")
	}
}

func (m *model) closeConsole() {
	m.console = false
	m.consoleIn.Blur()
}

func (m model) consoleView() string {
	gray := lipgloss.NewStyle().Foreground(colorGray)
	w := m.width - 4
	if w < 10 {
		w = 10
	}
	rule := gray.Render(strings.Repeat("─", w))
	title := aiLabelStyle.Render("Console") + gray.Render(" • "+backendLabel(m.backend)+" • "+m.currentModelName())
	hint := gray.Render(":help for commands • Up/Down command history • PgUp/PgDn scroll • Esc or Ctrl+G closes")
	return m.pageStyle().Render(lipgloss.JoinVertical(lipgloss.Left, title, rule, m.consoleVP.View(), rule, m.consoleIn.View(), hint))
}

// updateConsoleKey handles one key press while the console is open.
func (m *model) updateConsoleKey(msg tea.KeyMsg) tea.Cmd {
	switch msg.String() {
	case "esc", "ctrl+g":
		m.closeConsole()
		return nil
	case "enter":
		line := strings.TrimSpace(m.consoleIn.Value())
		m.consoleIn.SetValue("")
		if line == "" {
			return nil
		}
		m.consoleHist = capLines(append(m.consoleHist, line), 100)
		m.consoleHistPos = len(m.consoleHist)
		m.consolePrint("> " + line)
		return m.runConsole(line)
	case "up":
		if m.consoleHistPos > 0 {
			m.consoleHistPos--
			m.consoleIn.SetValue(m.consoleHist[m.consoleHistPos])
			m.consoleIn.CursorEnd()
		}
		return nil
	case "down":
		if m.consoleHistPos < len(m.consoleHist)-1 {
			m.consoleHistPos++
			m.consoleIn.SetValue(m.consoleHist[m.consoleHistPos])
			m.consoleIn.CursorEnd()
		} else {
			m.consoleHistPos = len(m.consoleHist)
			m.consoleIn.SetValue("")
		}
		return nil
	case "pgup":
		m.consoleVP.LineUp(m.consoleVP.Height / 2)
		return nil
	case "pgdown":
		m.consoleVP.LineDown(m.consoleVP.Height / 2)
		return nil
	}
	var cmd tea.Cmd
	m.consoleIn, cmd = m.consoleIn.Update(msg)
	return cmd
}

func (m *model) runConsole(line string) tea.Cmd {
	cmd, arg := parseConsole(line)
	switch cmd {
	case "":
	case "help", "h", "?":
		m.consolePrint(consoleHelp()...)
	case "exit", "quit", "q", "back":
		m.closeConsole()
	case "clear", "cls":
		m.consoleLines = nil
		m.refreshConsole()
	case "status":
		m.consolePrint(m.statusLines()...)
	case "tokens", "usage":
		m.consolePrint(m.tokenLines()...)
	case "stop":
		switch {
		case m.stopGeneration():
			m.consolePrint("✓ Stopped the reply.")
		case m.cancelConsoleChat():
			m.consolePrint("✓ Stopped the :chat reply.")
		default:
			m.consolePrint("Nothing is being generated right now.")
		}
	case "switch", "model":
		return m.consoleSwitch(arg)
	case "models", "ls":
		return m.consoleModels(arg)
	case "show", "info":
		return m.consoleShow(arg)
	case "ps":
		if m.isCloud() {
			m.consolePrint("✗ :ps only works with local Ollama (press Ctrl+B on the model screen to switch).")
			return nil
		}
		return consolePS(m.host)
	case "unload":
		if m.isCloud() {
			m.consolePrint("✗ :unload only works with local Ollama (press Ctrl+B on the model screen to switch).")
			return nil
		}
		return consoleUnload(m.host, arg)
	case "kill":
		return m.consoleKill()
	case "start":
		return consoleStart(m.host)
	case "chat", "ask":
		if arg == "" { // no prompt: show the whole chat instead
			m.consolePrint(m.dumpLines(dumpOptions{bodies: true})...)
			return nil
		}
		return m.consoleChat(arg)
	case "dump":
		m.consolePrint(m.dumpLines(dumpOptions{bodies: true, full: strings.EqualFold(arg, "full")})...)
	case "ctx", "context":
		m.consolePrint(m.dumpLines(dumpOptions{})...)
	case "colors", "colours", "color":
		m.consoleColors(arg)
	case "images", "imgs":
		m.consoleImages()
	case "view", "img", "image":
		return m.consoleViewImage(arg)
	case "copy":
		m.consoleCopy()
	case "export":
		m.consoleExport(arg)
	case "set":
		m.consoleSet(arg)
	case "system", "sys":
		m.consoleSystem(arg)
	case "top":
		return consoleTop()
	case "save":
		m.consoleSave(arg)
	case "logs", "path":
		m.consoleLogs()
	case "llama":
		m.consolePrint("cpp")
	default:
		m.consolePrint(fmt.Sprintf("✗ Unknown command %q. Type :help.", cmd))
	}
	return nil
}

func (m model) statusLines() []string {
	used, pct, _ := m.tokenMeter()
	lines := []string{
		"Backend:    " + backendLabel(m.backend),
		"Server:     " + m.endpointLabel(),
		"Model:      " + m.currentModelName(),
		fmt.Sprintf("Abilities:  vision=%v  thinking=%v  tools=%v", m.supportsVision, m.supportsThinking, m.supportsTools),
		fmt.Sprintf("Context:    ~%d / %d tokens (%.0f%%)", used, m.effectiveCtx(), pct),
		fmt.Sprintf("Chat:       %d messages", len(m.messages)),
	}
	switch {
	case m.isCloud() && m.apiKey() != "":
		lines = append(lines, "API key:    set")
	case m.isCloud():
		lines = append(lines, "API key:    missing (press Ctrl+K on the model screen)")
	case m.ollamaCmd != nil && m.ollamaCmd.Process != nil:
		lines = append(lines, fmt.Sprintf("Ollama:     started by ollama-tui (pid %d)", m.ollamaCmd.Process.Pid))
	default:
		lines = append(lines, "Ollama:     not started by ollama-tui")
	}
	if m.isThinking {
		lines = append(lines, "Reply:      being generated...")
	}
	return lines
}

func (m model) tokenLines() []string {
	used, pct, _ := m.tokenMeter()
	lines := []string{}
	if m.usageReplies == 0 {
		lines = append(lines, "No token counts yet: send a message first.")
	} else {
		lines = append(lines,
			"Token usage in this run (as reported by the server):",
			fmt.Sprintf("  model calls:       %d", m.usageReplies),
			fmt.Sprintf("  prompt tokens:     %d", m.usagePrompt),
			fmt.Sprintf("  generated tokens:  %d", m.usageCompletion),
			fmt.Sprintf("  total:             %d", m.usagePrompt+m.usageCompletion),
		)
		last := fmt.Sprintf("Last reply: %d prompt + %d generated tokens", m.lastUsage.Prompt, m.lastUsage.Completion)
		if m.lastUsage.TokensPerSec > 0 {
			last += fmt.Sprintf(", %.1f tokens/s", m.lastUsage.TokensPerSec)
		}
		lines = append(lines, last)
	}
	lines = append(lines, fmt.Sprintf("Context window: ~%d / %d tokens (%.0f%%), estimated at 4 characters per token", used, m.effectiveCtx(), pct))
	return lines
}

func (m *model) consoleSwitch(arg string) tea.Cmd {
	idx, err := resolveModel(arg, m.models)
	if err != nil {
		m.consolePrint("✗ " + err.Error())
		return nil
	}
	if m.isThinking {
		m.consolePrint("✗ A reply is being generated. Use :stop first.")
		return nil
	}
	name := m.models[idx]
	if m.isCloud() && m.apiKey() == "" {
		m.consolePrint("✗ No " + backendLabel(m.backend) + " API key yet (press Ctrl+K on the model screen).")
		return nil
	}

	m.cursor = idx
	m.filter = ""
	m.filterMode = false
	m.filterInput.SetValue("")
	note := "Switched model to: " + name
	if len(m.messages) == 0 {
		note = "Loaded model: " + name
	}
	m.messages = append(m.messages, ChatMessage{Role: "System", Content: note})
	if m.state == stateSelectModel { // like pressing Enter on the model screen
		m.state = stateChat
		m.focusMode = "textarea"
		m.textarea.Focus()
	}
	m.isSwitchingModel = false
	m.updateViewport()
	m.consolePrint("✓ " + note)

	if m.isCloud() {
		m.applyCaps(m.cloudCaps(name))
		return nil
	}
	return fetchCapabilities(m.host, name)
}

func (m *model) consoleModels(filter string) tea.Cmd {
	filter = strings.ToLower(strings.TrimSpace(filter))

	if m.isCloud() {
		var lines []string
		shown := 0
		for _, id := range m.models {
			if filter != "" && !strings.Contains(strings.ToLower(id), filter) {
				continue
			}
			shown++
			if len(lines) < 100 {
				info := m.cloudInfo[id]
				abilities := ""
				if info.Vision {
					abilities += " vision"
				}
				if info.Tools {
					abilities += " tools"
				}
				if info.Thinking {
					abilities += " reasoning"
				}
				lines = append(lines, fmt.Sprintf("  %-48s %8d ctx %s", id, info.Context, abilities))
			}
		}
		m.consolePrint(fmt.Sprintf("%d models (%s):", shown, backendLabel(m.backend)))
		m.consolePrint(lines...)
		if shown > len(lines) {
			m.consolePrint(fmt.Sprintf("  ... and %d more. Narrow it down with :models <text>", shown-len(lines)))
		}
		return nil
	}

	host := m.host
	return func() tea.Msg {
		var tags struct {
			Models []struct {
				Name    string `json:"name"`
				Size    int64  `json:"size"`
				Details struct {
					ParameterSize     string `json:"parameter_size"`
					QuantizationLevel string `json:"quantization_level"`
				} `json:"details"`
			} `json:"models"`
		}
		if err := httpJSON("GET", host+"/api/tags", nil, &tags); err != nil {
			return consoleOutMsg{[]string{"✗ " + err.Error()}}
		}
		lines := []string{}
		for _, t := range tags.Models {
			if filter != "" && !strings.Contains(strings.ToLower(t.Name), filter) {
				continue
			}
			lines = append(lines, fmt.Sprintf("  %-34s %9s  %7s  %s", t.Name, humanBytes(t.Size), t.Details.ParameterSize, t.Details.QuantizationLevel))
		}
		out := []string{fmt.Sprintf("%d models (Ollama):", len(lines))}
		return consoleOutMsg{append(out, lines...)}
	}
}

func (m *model) consoleShow(arg string) tea.Cmd {
	name := m.currentModelName()
	if arg != "" {
		idx, err := resolveModel(arg, m.models)
		if err != nil {
			m.consolePrint("✗ " + err.Error())
			return nil
		}
		name = m.models[idx]
	}
	if name == "(none)" {
		m.consolePrint("✗ No model selected yet.")
		return nil
	}

	if m.isCloud() {
		info := m.cloudInfo[name]
		m.consolePrint(
			name,
			fmt.Sprintf("  context:     %d tokens", info.Context),
			fmt.Sprintf("  vision:      %v", info.Vision),
			fmt.Sprintf("  tools:       %v", info.Tools),
			fmt.Sprintf("  reasoning:   %v", info.Thinking),
		)
		return nil
	}

	host := m.host
	return func() tea.Msg {
		var show struct {
			Details struct {
				Format            string `json:"format"`
				Family            string `json:"family"`
				ParameterSize     string `json:"parameter_size"`
				QuantizationLevel string `json:"quantization_level"`
			} `json:"details"`
			ModelInfo    map[string]interface{} `json:"model_info"`
			Capabilities []string               `json:"capabilities"`
		}
		body := map[string]string{"model": name, "name": name}
		if err := httpJSON("POST", host+"/api/show", body, &show); err != nil {
			return consoleOutMsg{[]string{"✗ " + err.Error()}}
		}
		lines := []string{
			name,
			"  family:        " + show.Details.Family,
			"  parameters:    " + show.Details.ParameterSize,
			"  quantization:  " + show.Details.QuantizationLevel,
			"  format:        " + show.Details.Format,
		}
		if ctx := contextLengthFromInfo(show.ModelInfo); ctx > 0 {
			lines = append(lines, fmt.Sprintf("  context:       %d tokens", ctx))
		}
		if len(show.Capabilities) > 0 {
			lines = append(lines, "  capabilities:  "+strings.Join(show.Capabilities, ", "))
		}
		return consoleOutMsg{lines}
	}
}

// httpJSON makes a short API call and decodes the JSON answer into out (if given).
func httpJSON(method, url string, body, out interface{}) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, url, rd)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := apiClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("%s: %s", resp.Status, apiErrorDetail(b))
	}
	if out == nil {
		io.Copy(io.Discard, resp.Body)
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type runningModel struct {
	Name      string    `json:"name"`
	Size      int64     `json:"size"`
	SizeVRAM  int64     `json:"size_vram"`
	ExpiresAt time.Time `json:"expires_at"`
}

func listRunning(host string) ([]runningModel, error) {
	var res struct {
		Models []runningModel `json:"models"`
	}
	if err := httpJSON("GET", host+"/api/ps", nil, &res); err != nil {
		return nil, err
	}
	return res.Models, nil
}

func consolePS(host string) tea.Cmd {
	return func() tea.Msg {
		models, err := listRunning(host)
		if err != nil {
			return consoleOutMsg{[]string{"✗ " + err.Error()}}
		}
		if len(models) == 0 {
			return consoleOutMsg{[]string{"No model is loaded in memory."}}
		}
		lines := []string{fmt.Sprintf("%d model(s) loaded:", len(models))}
		for _, r := range models {
			where := "CPU"
			if r.Size > 0 && r.SizeVRAM > 0 {
				where = fmt.Sprintf("%.0f%% in GPU memory", float64(r.SizeVRAM)/float64(r.Size)*100)
			}
			expires := ""
			if !r.ExpiresAt.IsZero() {
				if left := time.Until(r.ExpiresAt); left > 0 {
					expires = "  unloads in " + formatDuration(left)
				}
			}
			lines = append(lines, fmt.Sprintf("  %-30s %9s  %s%s", r.Name, humanBytes(r.Size), where, expires))
		}
		return consoleOutMsg{lines}
	}
}

func consoleUnload(host, arg string) tea.Cmd {
	return func() tea.Msg {
		models, err := listRunning(host)
		if err != nil {
			return consoleOutMsg{[]string{"✗ " + err.Error()}}
		}
		var lines []string
		for _, r := range models {
			if arg != "" && !strings.Contains(strings.ToLower(r.Name), strings.ToLower(arg)) {
				continue
			}
			body := map[string]interface{}{"model": r.Name, "keep_alive": 0, "stream": false}
			if err := httpJSON("POST", host+"/api/generate", body, nil); err != nil {
				lines = append(lines, "✗ "+r.Name+": "+err.Error())
			} else {
				lines = append(lines, "✓ Unloaded "+r.Name)
			}
		}
		if len(lines) == 0 {
			lines = []string{"No loaded model matches."}
		}
		return consoleOutMsg{lines}
	}
}

func (m *model) consoleKill() tea.Cmd {
	if m.ollamaCmd == nil {
		m.consolePrint("ollama-tui did not start the Ollama server, so it will not stop it.", "Use :unload to free the memory of the loaded models instead.")
		return nil
	}
	server := m.ollamaCmd
	m.consolePrint("Stopping the Ollama server...")
	return func() tea.Msg {
		stopOllama(server)
		return consoleKilledMsg{[]string{"✓ Ollama server stopped. Ctrl+R on the model screen starts it again."}}
	}
}

// --- Console: one-off questions, starting Ollama, saved logs ---

type consoleChatChunkMsg struct {
	ch       chan OllamaGenerateRes
	Response string
	Thinking string
	Error    string
	Usage    *UsageInfo
}
type consoleChatDoneMsg struct{ ch chan OllamaGenerateRes }

// consoleStartedMsg is the result of :start.
type consoleStartedMsg struct {
	cmd   *exec.Cmd // set if we had to start the server
	ready bool
	lines []string
}

func waitForConsoleChat(ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		res, ok := <-ch
		if !ok {
			return consoleChatDoneMsg{ch: ch}
		}
		return consoleChatChunkMsg{ch: ch, Response: res.Response, Thinking: res.Thinking, Error: res.Error, Usage: res.Usage}
	}
}

// consoleChat asks the current model a single question, without touching the
// chat, and reports token counts and speed when the answer is complete.
func (m *model) consoleChat(prompt string) tea.Cmd {
	prompt = strings.TrimSpace(prompt)
	switch {
	case prompt == "":
		m.consolePrint("usage: :chat <your question>")
		return nil
	case m.isThinking:
		m.consolePrint("✗ A reply is being generated in the chat. Use :stop first.")
		return nil
	case m.ccChan != nil:
		m.consolePrint("✗ Another :chat reply is still running (:stop cancels it).")
		return nil
	case len(m.models) == 0 || m.cursor < 0 || m.cursor >= len(m.models):
		m.consolePrint("✗ No model selected yet. Use :switch <name> or pick one on the model screen.")
		return nil
	case m.isCloud() && m.apiKey() == "":
		m.consolePrint("✗ No " + backendLabel(m.backend) + " API key yet (press Ctrl+K on the model screen).")
		return nil
	}

	name := m.models[m.cursor]
	msgs := buildChatMessages([]ChatMessage{{Role: "You", Content: prompt}}, m.systemPrompt, nil)
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan OllamaGenerateRes)
	m.ccChan, m.ccCancel = ch, cancel
	m.ccPrompt, m.ccText, m.ccThinking, m.ccTokens = prompt, "", "", nil
	m.ccStart, m.ccFirst, m.ccUsage = time.Now(), time.Time{}, nil
	m.consolePrint(fmt.Sprintf("Asking %s (%s)...", name, backendLabel(m.backend)))

	var gen tea.Cmd
	if m.isCloud() {
		gen = startGenerateCloud(ctx, m.backend, m.apiKey(), name, msgs, buildOptions(m.config, 0), nil, ch)
	} else {
		gen = startGenerate(ctx, m.host, name, msgs, nil, buildOptions(m.config, m.numCtxSetting()), nil, ch)
	}
	return tea.Batch(gen, waitForConsoleChat(ch))
}

// endConsoleChat lets go of a :chat request without printing its result.
func (m *model) endConsoleChat() {
	if m.ccCancel != nil {
		m.ccCancel()
	}
	m.ccChan, m.ccCancel = nil, nil
	m.ccText, m.ccThinking, m.ccTokens = "", "", nil
	m.refreshConsole()
}

func (m *model) cancelConsoleChat() bool {
	if m.ccChan == nil {
		return false
	}
	m.endConsoleChat()
	return true
}

// finishConsoleChat prints the answer of a :chat prompt and its statistics.
func (m *model) finishConsoleChat() {
	total := time.Since(m.ccStart)
	text, thinking, u := m.ccText, m.ccThinking, m.ccUsage

	promptTokens, completion := 0, 0
	if u != nil {
		promptTokens, completion = u.Prompt, u.Completion
	}
	estimated := false
	if completion == 0 {
		completion = (len(text) + len(thinking)) / 4
		estimated = true
	}
	if promptTokens == 0 {
		promptTokens = (len(m.ccPrompt) + len(m.systemPrompt)) / 4
		estimated = true
	}

	speed := 0.0
	if u != nil && u.TokensPerSec > 0 {
		speed = u.TokensPerSec
	} else if !m.ccFirst.IsZero() {
		if gen := time.Since(m.ccFirst).Seconds(); gen > 0 {
			speed = float64(completion) / gen
		}
	}

	// What was sent, then the reply with one colour per token.
	sent := buildChatMessages([]ChatMessage{{Role: "You", Content: m.ccPrompt}}, m.systemPrompt, nil)
	lines := requestBlockLines(sent, m.plainTokens)
	lines = append(lines, "", "AI:")
	if strings.TrimSpace(thinking) != "" {
		lines = append(lines, fmt.Sprintf("  (thought for %d characters first)", len(thinking)))
	}
	toks := m.ccTokens
	exact := len(toks) > 0
	if !exact {
		toks = splitTokens(text)
	}
	painted, _ := paintTokens(toks, 0, m.plainTokens)
	body := trimBlankLines(strings.Split(painted, "\n"))
	if len(body) == 0 {
		body = []string{"(empty reply)"}
	}
	lines = append(lines, indentLines(body, "  ")...)
	if !exact {
		lines = append(lines, "  (the server sent no separate tokens, so the colours are a rough guess)")
	}
	mark := ""
	if estimated {
		mark = "~"
	}
	lines = append(lines,
		"",
		"Reply stats:",
		"  model:             "+m.currentModelName()+" ("+backendLabel(m.backend)+")",
		fmt.Sprintf("  prompt tokens:     %s%d", mark, promptTokens),
		fmt.Sprintf("  generated tokens:  %s%d", mark, completion),
	)
	if speed > 0 {
		lines = append(lines, fmt.Sprintf("  speed:             %.1f tokens/s", speed))
	}
	if !m.ccFirst.IsZero() {
		lines = append(lines, "  first token after: "+formatDuration(m.ccFirst.Sub(m.ccStart)))
	}
	lines = append(lines, "  total time:        "+formatDuration(total))
	if estimated {
		lines = append(lines, "  (~ means estimated at 4 characters per token: the server gave no count)")
	}

	if u != nil {
		m.addUsage(*u)
	} else {
		m.addUsage(UsageInfo{Prompt: promptTokens, Completion: completion, TokensPerSec: speed})
	}
	if m.ccCancel != nil {
		m.ccCancel()
	}
	m.ccChan, m.ccCancel = nil, nil // before printing, so the live copy disappears
	m.ccText, m.ccThinking, m.ccTokens = "", "", nil
	m.consolePrint(lines...)
}

// consoleStart starts the local Ollama server (like launching Ollama-TUI would).
func consoleStart(host string) tea.Cmd {
	return func() tea.Msg {
		if host != defaultHost {
			return consoleStartedMsg{lines: []string{"✗ I can only start a server on this computer (the current host is " + host + ")."}}
		}
		if ollamaReachable(host) {
			return consoleStartedMsg{ready: true, lines: []string{"✓ Ollama is already running at " + host + "."}}
		}
		path, err := exec.LookPath("ollama")
		if err != nil {
			return consoleStartedMsg{lines: []string{"✗ The ollama command was not found in your PATH. Install Ollama first."}}
		}
		cmd := exec.Command(path, "serve")
		if err := cmd.Start(); err != nil {
			return consoleStartedMsg{lines: []string{"✗ Could not start Ollama: " + err.Error()}}
		}
		for i := 0; i < 20 && !ollamaReachable(host); i++ {
			time.Sleep(250 * time.Millisecond)
		}
		if !ollamaReachable(host) {
			return consoleStartedMsg{cmd: cmd, lines: []string{"Started `ollama serve`, but it is not answering yet. Try :models in a moment."}}
		}
		return consoleStartedMsg{cmd: cmd, ready: true, lines: []string{fmt.Sprintf("✓ Started the Ollama server (pid %d). :kill stops it again.", cmd.Process.Pid)}}
	}
}

// consoleTop prints the load of this computer, with bars.
func consoleTop() tea.Cmd {
	return func() tea.Msg {
		_, first := readSystemStats(cpuTimes{}, false) // the CPU load needs two readings
		time.Sleep(300 * time.Millisecond)
		stats, _ := readSystemStats(first, true)
		lines := append([]string{"System load right now:"}, stats.Lines(24)...)
		return consoleOutMsg{lines}
	}
}

// --- Saved console logs ---

type consoleLogFile struct {
	SavedAt string `json:"saved_at"`
	Backend string `json:"backend"`
	Model   string `json:"model"`
	Usage   struct {
		ModelCalls      int `json:"model_calls"`
		PromptTokens    int `json:"prompt_tokens"`
		GeneratedTokens int `json:"generated_tokens"`
	} `json:"usage"`
	Lines []string `json:"lines"`
}

// sanitizeFileName keeps letters, digits, - _ and . and turns the rest into _.
func sanitizeFileName(s string) string {
	var b strings.Builder
	for _, r := range strings.TrimSpace(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.TrimLeft(b.String(), ".")
	if len(out) > 60 {
		out = out[:60]
	}
	return out
}

// consoleSave writes the console log to a .json file in the console folder.
func (m *model) consoleSave(name string) {
	dir := consoleLogDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		m.consolePrint("✗ " + err.Error())
		return
	}
	file := sanitizeFileName(name)
	if file == "" {
		file = "console_" + time.Now().Format("20060102_150405")
	}
	if !strings.HasSuffix(file, ".json") {
		file += ".json"
	}

	var log consoleLogFile
	log.SavedAt = time.Now().Format(time.RFC3339)
	log.Backend = backendLabel(m.backend)
	log.Model = m.currentModelName()
	log.Usage.ModelCalls = m.usageReplies
	log.Usage.PromptTokens = m.usagePrompt
	log.Usage.GeneratedTokens = m.usageCompletion
	log.Lines = stripANSILines(m.consoleLines)

	data, err := json.MarshalIndent(log, "", "  ")
	if err != nil {
		m.consolePrint("✗ " + err.Error())
		return
	}
	path := filepath.Join(dir, file)
	if err := os.WriteFile(path, data, 0600); err != nil {
		m.consolePrint("✗ " + err.Error())
		return
	}
	m.consolePrint("✓ Saved the console log to " + path)
}

// recentFiles returns the names of the newest .json files in dir (at most n).
func recentFiles(dir string, n int) []string {
	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	names := make([]string, 0, len(files))
	for _, f := range files {
		names = append(names, filepath.Base(f))
	}
	sort.Sort(sort.Reverse(sort.StringSlice(names)))
	if len(names) > n {
		names = names[:n]
	}
	return names
}

func (m *model) consoleLogs() {
	lines := []string{"Saved chats:        " + historyDir()}
	for _, f := range recentFiles(historyDir(), 5) {
		lines = append(lines, "  "+f)
	}
	lines = append(lines, "Saved console logs: "+consoleLogDir())
	for _, f := range recentFiles(consoleLogDir(), 5) {
		lines = append(lines, "  "+f)
	}
	lines = append(lines, "(:save [name] writes a console log, Shift+J in the chat saves the chat)")
	m.consolePrint(lines...)
}

// --- A hidden key sequence on the settings screen ---

var konamiCode = []string{"up", "up", "down", "down", "left", "right", "left", "right", "b", "a"}

// konamiStep feeds one key to the sequence tracker. swallow means the key was
// one of the letters and must not be typed into a field; open means the whole
// sequence and Enter were entered.
func konamiStep(progress int, key string) (next int, swallow, open bool) {
	if progress == len(konamiCode) {
		if key == "enter" {
			return 0, true, true
		}
		progress = 0
	}
	if key == konamiCode[progress] {
		progress++
		return progress, key == "b" || key == "a", false
	}
	if key == konamiCode[0] {
		return 1, false, false
	}
	return 0, false, false
}
