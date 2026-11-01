package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/atotto/clipboard"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ledongthuc/pdf"
)

// --- Ollama API Structs ---
type OllamaModel struct {
	Name string `json:"name"`
}
type OllamaTags struct {
	Models []OllamaModel `json:"models"`
}
type OllamaGenerateReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"`
	Stream bool   `json:"stream"`
}
type OllamaGenerateRes struct {
	Response string `json:"response"`
}

// --- Config & App Models ---
type Config struct {
	SystemPrompt string `json:"system_prompt"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Attachment struct {
	Name string
	Path string
	Text string
}

type state int

const (
	stateLoading state = iota
	stateSelectModel
	stateChat
	stateHistory
	stateSystemPrompt
	stateFileAttach
	stateAttachmentList
	stateError
	stateExiting
)

// --- UI Styles ---
var (
	colorPurple = lipgloss.Color("135")
	colorCyan   = lipgloss.Color("81")
	colorGray   = lipgloss.Color("240")
	colorGreen  = lipgloss.Color("42")
	colorRed    = lipgloss.Color("196")
	colorYellow = lipgloss.Color("220")

	userLabelStyle = lipgloss.NewStyle().Foreground(colorCyan).Bold(true)
	aiLabelStyle   = lipgloss.NewStyle().Foreground(colorPurple).Bold(true)

	headerStyle   = lipgloss.NewStyle().Foreground(colorGray).Border(lipgloss.NormalBorder(), false, false, true, false).Padding(0, 1)
	textareaStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(colorGray).Padding(0, 1)
	statusStyle   = lipgloss.NewStyle().Foreground(colorGray).Padding(0, 1)
)

type model struct {
	state            state
	models           []string
	cursor           int
	messages         []ChatMessage
	selectedMsg      int
	focusMode        string
	statusMsg        string
	systemPrompt     string
	config           Config
	viewport         viewport.Model
	textarea         textarea.Model
	promptInput      textarea.Model
	fileInput        textarea.Model
	attachments      []Attachment
	attachCursor     int
	swapIndex        int // -1 for adding new, >=0 for swapping
	ollamaCmd        *exec.Cmd
	errText          string
	keepServer       bool
	err              error
	streamChan       chan string
	cancelStream     context.CancelFunc
	isThinking       bool
	autoScroll       bool
	isSwitchingModel bool
	historyFiles     []string
	historyCursor    int
	width            int
	height           int
}

// --- Config Helpers ---
func getConfigPath() string {
	configDir, _ := os.UserConfigDir()
	return filepath.Join(configDir, "ollama-tui", "config.json")
}

func loadConfig() Config {
	var cfg Config
	data, err := os.ReadFile(getConfigPath())
	if err == nil {
		json.Unmarshal(data, &cfg)
	}
	return cfg
}

func saveConfig(cfg Config) {
	path := getConfigPath()
	os.MkdirAll(filepath.Dir(path), 0755)
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(path, data, 0644)
}

// --- Custom Messages ---
type modelsFetchedMsg struct{ models []string }
type errMsg struct{ err error }
func (e errMsg) Error() string { return e.err.Error() }
type clearStatusMsg struct{}
type historyFilesMsg struct{ files []string }
type ollamaMissingMsg struct{}
type noModelsMsg struct{}
type fileAttachedMsg struct{ attachment Attachment }
type fileErrorMsg struct{ err string }
type fileEditedMsg struct{ index int; text string }

// Init runs when the app starts
func (m model) Init() tea.Cmd {
	return func() tea.Msg {
		if m.ollamaCmd == nil {
			return ollamaMissingMsg{}
		}

		resp, err := http.Get("http://localhost:11434/api/tags")
		if err != nil {
			return ollamaMissingMsg{}
		}
		defer resp.Body.Close()

		var tags OllamaTags
		json.NewDecoder(resp.Body).Decode(&tags)

		if len(tags.Models) == 0 {
			return noModelsMsg{}
		}

		var modelNames []string
		for _, m := range tags.Models {
			modelNames = append(modelNames, m.Name)
		}

		return modelsFetchedMsg{models: modelNames}
	}
}

func waitForStream(ch chan string) tea.Cmd {
	return func() tea.Msg {
		text, ok := <-ch
		if !ok { return streamDoneMsg{} }
		return streamChunkMsg(text)
	}
}

type streamChunkMsg string
type streamDoneMsg struct{}

func startGenerate(ctx context.Context, modelName, prompt, system string, ch chan string) tea.Cmd {
	return func() tea.Msg {
		reqBody, _ := json.Marshal(OllamaGenerateReq{
			Model:  modelName,
			Prompt: prompt,
			System: system,
			Stream: true,
		})

		req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:11434/api/generate", bytes.NewBuffer(reqBody))
		if err != nil { close(ch); return nil }
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil { close(ch); return nil }
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			var res OllamaGenerateRes
			json.Unmarshal(scanner.Bytes(), &res)
			if res.Response != "" { ch <- res.Response }
		}
		close(ch)
		return nil
	}
}

func loadHistoryFiles() tea.Msg {
	dir, _ := os.UserConfigDir()
	histDir := filepath.Join(filepath.Dir(dir), ".local/share/ollama-tui")
	if _, err := os.Stat(histDir); os.IsNotExist(err) {
		histDir = filepath.Join(os.Getenv("HOME"), ".local/share/ollama-tui")
	}
	files, _ := filepath.Glob(histDir + "/*.json")
	sort.Sort(sort.Reverse(sort.StringSlice(files)))
	return historyFilesMsg{files: files}
}

// --- File Parsing Helpers ---
func readTextFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	return string(data), err
}

func readPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil { return "", err }
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil { return "", err }
	buf.ReadFrom(b)
	return buf.String(), nil
}

func extractXMLFromZip(path, targetFile string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil { return "", err }
	defer r.Close()

	for _, f := range r.File {
		if f.Name == targetFile {
			rc, err := f.Open()
			if err != nil { return "", err }
			defer rc.Close()
			data, err := io.ReadAll(rc)
			return string(data), err
		}
	}
	return "", fmt.Errorf("could not find %s in %s", targetFile, path)
}

func stripXML(xmlContent string) string {
	decoder := xml.NewDecoder(strings.NewReader(xmlContent))
	var textBuilder strings.Builder

	for {
		token, err := decoder.Token()
		if err != nil {
			break
		}
		// We only care about the actual text, not the tags
		switch t := token.(type) {
			case xml.CharData:
				text := strings.TrimSpace(string(t))
				if text != "" {
					textBuilder.WriteString(text + " ")
				}
		}
	}
	return textBuilder.String()
}

func processFile(path string) tea.Msg {
	path = strings.TrimSpace(path)
	path = strings.Trim(path, `"'`)
	ext := strings.ToLower(filepath.Ext(path))
	var content string
	var err error

	switch ext {
		case ".txt", ".md", ".go", ".c", ".h", ".py", ".js", ".json", ".sh", ".yaml", ".yml", ".rs", ".toml", ".ini", ".conf":
			content, err = readTextFile(path)
		case ".pdf":
			content, err = readPDF(path)
		case ".docx":
			var xmlStr string
			xmlStr, err = extractXMLFromZip(path, "word/document.xml")
			if err == nil { content = stripXML(xmlStr) }
		case ".odt":
			var xmlStr string
			xmlStr, err = extractXMLFromZip(path, "content.xml")
			if err == nil { content = stripXML(xmlStr) }
		case ".abw":
			var xmlStr string
			xmlStr, err = readTextFile(path)
			if err == nil { content = stripXML(xmlStr) }
		default:
			return fileErrorMsg{err: fmt.Sprintf("unsupported file type: %s", ext)}
	}

	if err != nil { return fileErrorMsg{err: err.Error()} }

	content = strings.Join(strings.Fields(content), " ")
	if len(content) > 20000 {
		content = content[:20000] + "\n...[TRUNCATED]..."
	}

	return fileAttachedMsg{attachment: Attachment{Name: filepath.Base(path), Path: path, Text: content}}
}

func editInNano(path string, index int) tea.Cmd {
	c := exec.Command("nano", path)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		if err != nil { return fileErrorMsg{err: err.Error()} }
		content, err := readTextFile(path)
		if err != nil { return fileErrorMsg{err: err.Error()} }
		content = strings.Join(strings.Fields(content), " ")
		return fileEditedMsg{index: index, text: content}
	})
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
		case tea.WindowSizeMsg:
			m.width = msg.Width
			m.height = msg.Height
			m.viewport.Width = msg.Width - 4
			m.viewport.Height = msg.Height - 10
			m.textarea.SetWidth(msg.Width - 6)
			m.promptInput.SetWidth(msg.Width - 6)
			m.fileInput.SetWidth(msg.Width - 10) // Account for borders and prompt
			m.updateViewport()

		case clearStatusMsg:
			m.statusMsg = ""

		case ollamaMissingMsg:
			m.state = stateError
			m.errText = "you need to install Ollama CLI tools\n\nGet it at: https://ollama.com/"
			m.keepServer = false

		case noModelsMsg:
			m.state = stateError
			m.errText = "please install a model with 'ollama pull <model name>'\n\nwe'll keep the server running when you exit\nuntil you install a model and relaunch."
			m.keepServer = true

		case fileAttachedMsg:
			if m.swapIndex == -1 {
				m.attachments = append(m.attachments, msg.attachment)
			} else {
				m.attachments[m.swapIndex] = msg.attachment
				m.swapIndex = -1
			}
			m.state = stateAttachmentList
			m.statusMsg = fmt.Sprintf("Attached: %s", msg.attachment.Name)
			cmds = append(cmds, func() tea.Msg { time.Sleep(2 * time.Second); return clearStatusMsg{} })

		case fileEditedMsg:
			m.attachments[msg.index].Text = msg.text
			m.state = stateAttachmentList
			m.statusMsg = "File edited and reloaded."
			cmds = append(cmds, func() tea.Msg { time.Sleep(2 * time.Second); return clearStatusMsg{} })

		case fileErrorMsg:
			m.state = stateChat
			m.focusMode = "textarea"
			m.textarea.Focus()
			m.statusMsg = "File error: " + msg.err
			cmds = append(cmds, func() tea.Msg { time.Sleep(4 * time.Second); return clearStatusMsg{} })

		case modelsFetchedMsg:
			m.models = msg.models
			m.state = stateSelectModel

		case historyFilesMsg:
			m.historyFiles = msg.files
			m.historyCursor = 0
			m.state = stateHistory

		case streamChunkMsg:
			m.messages[len(m.messages)-1].Content += string(msg)
			m.updateViewport()
			if m.autoScroll { m.viewport.GotoBottom() }
			cmds = append(cmds, waitForStream(m.streamChan))

		case streamDoneMsg:
			m.isThinking = false

		case errMsg:
			m.err = msg
			return m, tea.Quit

		case tea.KeyMsg:
			if m.state == stateError {
				if msg.String() == "ctrl+c" || msg.String() == "enter" || msg.String() == "esc" {
					if m.ollamaCmd != nil && m.ollamaCmd.Process != nil && !m.keepServer {
						m.ollamaCmd.Process.Kill()
					}
					return m, tea.Quit
				}
				return m, nil
			}

			if msg.String() == "ctrl+c" {
				if m.state == stateChat || m.state == stateHistory || m.state == stateSystemPrompt || m.state == stateFileAttach || m.state == stateAttachmentList {
					m.state = stateExiting
					m.textarea.Blur()
					m.promptInput.Blur()
					m.fileInput.Blur()
				} else {
					return m, tea.Quit
				}
			}

			if m.state == stateExiting {
				if msg.String() == "y" || msg.String() == "Y" {
					if m.ollamaCmd != nil && m.ollamaCmd.Process != nil {
						m.ollamaCmd.Process.Kill()
					}
					return m, tea.Quit
				}
				if msg.String() == "n" || msg.String() == "N" { return m, tea.Quit }
				return m, nil
			}

			if m.state == stateFileAttach {
				var cmd tea.Cmd
				m.fileInput, cmd = m.fileInput.Update(msg)
				cmds = append(cmds, cmd)

				if msg.String() == "esc" {
					m.state = stateChat
					m.focusMode = "textarea"
					m.fileInput.Blur()
					m.textarea.Focus()
					m.statusMsg = "Attachment cancelled."
				} else if msg.String() == "enter" {
					path := m.fileInput.Value()
					m.fileInput.Reset()
					m.fileInput.Blur()
					if path != "" {
						cmds = append(cmds, func() tea.Msg { return processFile(path) })
					} else {
						m.state = stateChat
						m.focusMode = "textarea"
						m.textarea.Focus()
					}
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateAttachmentList {
				switch msg.String() {
					case "esc", "ctrl+s":
						m.state = stateChat
						m.focusMode = "textarea"
						m.textarea.Focus()
					case "up", "k":
						if m.attachCursor > 0 { m.attachCursor-- }
					case "down", "j":
						if m.attachCursor < len(m.attachments)-1 { m.attachCursor++ }
					case "X": // Delete
						if len(m.attachments) > 0 {
							m.attachments = append(m.attachments[:m.attachCursor], m.attachments[m.attachCursor+1:]...)
							if m.attachCursor >= len(m.attachments) && m.attachCursor > 0 { m.attachCursor-- }
							m.statusMsg = "Attachment deleted."
						}
					case "S": // Swap
						if len(m.attachments) > 0 {
							m.swapIndex = m.attachCursor
							m.state = stateFileAttach
							m.fileInput.Focus()
						}
					case "N": // Edit in Nano
						if len(m.attachments) > 0 {
							attach := m.attachments[m.attachCursor]
							cmds = append(cmds, editInNano(attach.Path, m.attachCursor))
						}
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateSystemPrompt {
				var cmd tea.Cmd
				m.promptInput, cmd = m.promptInput.Update(msg)
				cmds = append(cmds, cmd)

				if msg.String() == "esc" {
					m.state = stateChat
					m.focusMode = "list"
					m.promptInput.Blur()
					m.statusMsg = "Prompt edit cancelled."
				} else if msg.String() == "ctrl+s" || msg.String() == "enter" {
					m.systemPrompt = m.promptInput.Value()
					m.config.SystemPrompt = m.systemPrompt
					saveConfig(m.config)
					m.state = stateChat
					m.focusMode = "list"
					m.promptInput.Blur()
					m.statusMsg = "System prompt saved."
					cmds = append(cmds, func() tea.Msg { time.Sleep(2 * time.Second); return clearStatusMsg{} })
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateHistory {
				switch msg.String() {
					case "esc":
						m.state = stateChat
						m.focusMode = "list"
					case "up", "k":
						if m.historyCursor > 0 { m.historyCursor-- }
					case "down", "j":
						if m.historyCursor < len(m.historyFiles)-1 { m.historyCursor++ }
					case "enter":
						if len(m.historyFiles) > 0 {
							data, err := os.ReadFile(m.historyFiles[m.historyCursor])
							if err == nil {
								json.Unmarshal(data, &m.messages)
								m.state = stateChat
								m.focusMode = "list"
								m.selectedMsg = len(m.messages) - 1
								m.autoScroll = true
								m.updateViewport()
								m.viewport.GotoBottom()
								m.statusMsg = "Loaded history."
							}
						}
				}
				return m, nil
			}

			if m.state == stateSelectModel {
				switch msg.String() {
					case "up", "k":
						if m.cursor > 0 { m.cursor-- }
					case "down", "j":
						if m.cursor < len(m.models)-1 { m.cursor++ }
					case "enter":
						m.state = stateChat
						m.focusMode = "textarea"
						m.selectedMsg = -1
						m.autoScroll = true

						if m.isSwitchingModel {
							m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("Switched model to: %s", m.models[m.cursor])})
							m.isSwitchingModel = false
						} else if len(m.messages) == 0 {
							m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("Loaded model: %s", m.models[m.cursor])})
						}

						m.updateViewport()
						m.textarea.Focus()
				}
				return m, nil
			}

			if m.state == stateChat {
				if msg.String() == "alt+s" && m.isThinking {
					if m.cancelStream != nil { m.cancelStream() }
					m.isThinking = false
					m.messages[len(m.messages)-1].Content += " [Stopped by user]"
					m.updateViewport()
					return m, nil
				}

				if msg.String() == "tab" || msg.String() == "esc" {
					if m.focusMode == "textarea" {
						m.focusMode = "list"
						m.textarea.Blur()
						if m.selectedMsg == -1 { m.selectedMsg = len(m.messages) - 1 }
					} else {
						m.focusMode = "textarea"
						m.textarea.Focus()
					}
					m.updateViewport()
					return m, nil
				}

				if m.focusMode == "list" {
					switch msg.String() {
						case "up", "k":
							if m.selectedMsg > 0 { m.selectedMsg--; m.autoScroll = false; m.updateViewport() }
						case "down", "j":
							if m.selectedMsg < len(m.messages)-1 {
								m.selectedMsg++
								if m.selectedMsg == len(m.messages)-1 { m.autoScroll = true }
								m.updateViewport()
							}
						case "z", "w":
							m.viewport.LineUp(1)
							m.autoScroll = false
						case "s":
							m.viewport.LineDown(1)
						case "C":
							if m.selectedMsg >= 0 && m.selectedMsg < len(m.messages) {
								clipboard.WriteAll(m.messages[m.selectedMsg].Content)
								m.statusMsg = "Copied to clipboard!"
								cmds = append(cmds, func() tea.Msg { time.Sleep(2 * time.Second); return clearStatusMsg{} })
							}
						case "N":
							m.messages = []ChatMessage{{Role: "System", Content: fmt.Sprintf("Started new chat with %s", m.models[m.cursor])}}
							m.selectedMsg = 0
							m.autoScroll = true
							m.updateViewport()
							m.statusMsg = "Started new chat."
						case "S":
							m.state = stateSelectModel
							m.isSwitchingModel = true
						case "P":
							m.state = stateSystemPrompt
							m.promptInput.SetValue(m.systemPrompt)
							m.promptInput.Focus()
						case "J":
							os.MkdirAll(os.Getenv("HOME")+"/.local/share/ollama-tui", 0755)
							filename := fmt.Sprintf("chat_%s.json", time.Now().Format("20060102_150405"))
							path := os.Getenv("HOME") + "/.local/share/ollama-tui/" + filename
							data, _ := json.MarshalIndent(m.messages, "", "  ")
							os.WriteFile(path, data, 0644)
							m.statusMsg = "Saved: " + filename
							cmds = append(cmds, func() tea.Msg { time.Sleep(3 * time.Second); return clearStatusMsg{} })
						case "B":
							if m.selectedMsg >= 0 && m.selectedMsg < len(m.messages) {
								m.messages = m.messages[:m.selectedMsg+1]
								m.messages = append(m.messages, ChatMessage{Role: "System", Content: "--- Branched from here ---"})
								m.selectedMsg = len(m.messages) - 1
								m.autoScroll = true
								m.updateViewport()
								m.statusMsg = "Branched chat."
							}
						case "ctrl+o":
							return m, loadHistoryFiles
						case "ctrl+l":
							if len(m.attachments) > 0 {
								m.state = stateAttachmentList
								m.attachCursor = 0
							}
					}
					return m, tea.Batch(cmds...)
				}

				if m.focusMode == "textarea" && !m.isThinking {
					var cmd tea.Cmd

					if msg.String() == "ctrl+f" {
						m.state = stateFileAttach
						m.swapIndex = -1
						m.textarea.Blur()
						m.fileInput.Focus()
						return m, nil
					}
					if msg.String() == "ctrl+l" {
						if len(m.attachments) > 0 {
							m.state = stateAttachmentList
							m.attachCursor = 0
						}
						return m, nil
					}

					m.textarea, cmd = m.textarea.Update(msg)
					cmds = append(cmds, cmd)

					if msg.String() == "enter" && m.textarea.Value() != "" {
						userPrompt := m.textarea.Value()

						var contextBuilder strings.Builder
						if len(m.attachments) > 0 {
							contextBuilder.WriteString("Context from attached files:\n")
							for _, a := range m.attachments {
								contextBuilder.WriteString(fmt.Sprintf("--- %s ---\n%s\n\n", a.Name, a.Text))
							}
						}

						finalPrompt := userPrompt
						if len(m.attachments) > 0 {
							finalPrompt = contextBuilder.String() + "User Prompt: " + userPrompt
						}

						m.messages = append(m.messages, ChatMessage{Role: "You", Content: userPrompt})
						for _, a := range m.attachments {
							m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("📎 Attached: %s", a.Name)})
						}
						m.messages = append(m.messages, ChatMessage{Role: "AI", Content: ""})

						m.attachments = nil // Clear after sending

						m.selectedMsg = -1
						m.streamChan = make(chan string)
						m.isThinking = true
						m.autoScroll = true

						ctx, cancel := context.WithCancel(context.Background())
						m.cancelStream = cancel

						m.textarea.Reset()
						m.updateViewport()
						m.viewport.GotoBottom()

						cmds = append(cmds, startGenerate(ctx, m.models[m.cursor], finalPrompt, m.systemPrompt, m.streamChan))
						cmds = append(cmds, waitForStream(m.streamChan))
					}
				}
			}
	}

	return m, tea.Batch(cmds...)
}

func (m *model) updateViewport() {
	var rendered []string
	textWidth := m.viewport.Width - 3

	for i, msg := range m.messages {
		isSelected := (i == m.selectedMsg && m.focusMode == "list")

		var label string
		if msg.Role == "You" { label = "You:" } else if msg.Role == "AI" { label = "AI:" } else { label = "•" }

		pointer := "   "
		if isSelected { pointer = lipgloss.NewStyle().Foreground(colorCyan).Bold(true).Render(" > ") }

		var styledLabel string
		if isSelected {
			bg := colorCyan
			if msg.Role == "AI" { bg = colorPurple }
			if msg.Role == "System" { bg = colorGray }
			styledLabel = lipgloss.NewStyle().Background(bg).Foreground(lipgloss.Color("0")).Bold(true).Render(" " + label + " ")
		} else {
			if msg.Role == "You" { styledLabel = userLabelStyle.Render(label) } else if msg.Role == "AI" { styledLabel = aiLabelStyle.Render(label) } else { styledLabel = lipgloss.NewStyle().Foreground(colorGray).Italic(true).Render(label) }
		}

		labelWidth := lipgloss.Width(styledLabel)
		wrapWidth := textWidth - labelWidth - 1
		if wrapWidth < 10 { wrapWidth = 10 }

		wrappedText := lipgloss.NewStyle().Width(wrapWidth).Render(msg.Content)
		lines := strings.Split(wrappedText, "\n")

		var combined []string
		combined = append(combined, styledLabel+" "+lines[0])
		indent := strings.Repeat(" ", labelWidth+1)
		for j := 1; j < len(lines); j++ { combined = append(combined, indent+lines[j]) }

		block := strings.Join(combined, "\n")
		rendered = append(rendered, lipgloss.JoinHorizontal(lipgloss.Top, pointer, block))
	}

	m.viewport.SetContent(lipgloss.JoinVertical(lipgloss.Left, rendered...))
}
func (m model) View() string {
	if m.err != nil { return fmt.Sprintf("Error: %v\n", m.err) }

	if m.state == stateError {
		errBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(colorRed).
		Padding(1, 2).
		Render(lipgloss.NewStyle().Foreground(colorRed).Bold(true).Render("ERROR: ") + m.errText)
		return lipgloss.NewStyle().Margin(2, 4).Render(errBox)
	}

	if m.state == stateLoading { return "\n  Connecting to Ollama and fetching models...\n" }
	if m.state == stateExiting { return "\n  Do you want to stop the Ollama server? (y/n)\n" }

	if m.state == stateFileAttach {
		title := "Attach File (Ctrl+F)"
		if m.swapIndex != -1 { title = "Replace Attachment (Shift+S)" }

		// Make the box slightly narrower than the screen
		box := textareaStyle.Width(m.width - 10).Render(m.fileInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Drag & drop a file or type a path. Press Enter to attach, Esc to cancel.")

		content := lipgloss.JoinVertical(
			lipgloss.Left,
			aiLabelStyle.Render(title),
						 "",
				   box,
				   "",
				   help,
		)
		return lipgloss.NewStyle().Margin(1, 2).Render(content)
	}

	if m.state == stateAttachmentList {
		s := "Manage Attachments (Ctrl+L)\n\n"
		if len(m.attachments) == 0 {
			s += "No files attached.\nPress Esc to return."
		} else {
			for i, a := range m.attachments {
				cursor := " "
				style := lipgloss.NewStyle()
				if m.attachCursor == i { cursor = ">"; style = style.Foreground(colorCyan).Bold(true) }
				s += style.Render(fmt.Sprintf("%s %s (%d chars)", cursor, a.Name, len(a.Text))) + "\n"
			}
			s += "\n↑/↓ Select • Shift+X Delete • Shift+S Replace • Shift+N Edit in Nano • Esc/Ctrl+S Save & Return"
		}
		return lipgloss.NewStyle().Margin(1, 2).Render(s)
	}

	if m.state == stateSystemPrompt {
		box := textareaStyle.Width(m.width - 10).Render(m.promptInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Press Enter or Ctrl+S to save, Esc to cancel.")

		content := lipgloss.JoinVertical(
			lipgloss.Left,
			aiLabelStyle.Render("Edit System Prompt"),
						 "",
				   box,
				   "",
				   help,
		)
		return lipgloss.NewStyle().Margin(1, 2).Render(content)
	}

	if m.state == stateHistory {
		s := "Select a chat history to load (Up/Down, Enter to load, Esc to cancel):\n\n"
		if len(m.historyFiles) == 0 { s += "No saved histories found.\n" }
		for i, choice := range m.historyFiles {
			cursor := " "
			style := lipgloss.NewStyle()
			if m.historyCursor == i { cursor = ">"; style = style.Foreground(colorCyan).Bold(true) }
			s += style.Render(fmt.Sprintf("%s %s", cursor, filepath.Base(choice))) + "\n"
		}
		return lipgloss.NewStyle().Margin(1, 2).Render(s)
	}

	if m.state == stateSelectModel {
		s := "Select a model (Up/Down, Enter to select, Ctrl+C to quit):\n\n"
		for i, choice := range m.models {
			cursor := " "
			style := lipgloss.NewStyle()
			if m.cursor == i { cursor = ">"; style = style.Foreground(colorCyan).Bold(true) }
			s += style.Render(fmt.Sprintf("%s %s", cursor, choice)) + "\n"
		}
		return lipgloss.NewStyle().Margin(1, 2).Render(s)
	}

	// --- CHAT STATE UI ---
	modelName := m.models[m.cursor]
	totalChars := 0
	for _, msg := range m.messages { totalChars += len(msg.Content) }
	for _, a := range m.attachments { totalChars += len(a.Text) }

	approxTokens := totalChars / 4
	percent := (float64(approxTokens) / float64(4096)) * 100
	meterColor := colorGreen
	if percent > 75 { meterColor = colorYellow }
	if percent > 90 { meterColor = colorRed }
	meter := lipgloss.NewStyle().Foreground(meterColor).Render(fmt.Sprintf("[%.0f%% Context]", percent))
	header := headerStyle.Render(fmt.Sprintf("Model: %s  %s", aiLabelStyle.Render(modelName), meter))

	vpView := m.viewport.View()
	taView := textareaStyle.Render(m.textarea.View())

	var statusText string
	if m.statusMsg != "" {
		statusText = lipgloss.NewStyle().Foreground(colorGreen).Render(m.statusMsg)
	} else if m.isThinking {
		statusText = lipgloss.NewStyle().Foreground(colorRed).Render("Thinking... (Alt+S to stop)")
	} else if m.focusMode == "list" {
		statusText = lipgloss.NewStyle().Foreground(colorPurple).Render("[LIST] w/z/s Scroll • ↑/↓ Select • C Copy • N New • S Switch • P Prompt • J Save • B Branch • Ctrl+O Hist • Ctrl+L Attach")
	} else {
		if len(m.attachments) > 0 {
			statusText = lipgloss.NewStyle().Foreground(colorYellow).Bold(true).Render(fmt.Sprintf("[%d Attached] ", len(m.attachments))) + "[CHAT MODE] Ctrl+L to manage • Tab for list"
		} else {
			statusText = "[CHAT MODE] Ctrl+F to attach file • Ctrl+L to manage • Tab for list mode"
		}
	}
	status := statusStyle.Render(statusText)

	return lipgloss.JoinVertical(lipgloss.Left, header, vpView, taView, status)
}

func main() {
	cfg := loadConfig()

	var ollamaCmd *exec.Cmd
	if _, err := exec.LookPath("ollama"); err == nil {
		ollamaCmd = exec.Command("ollama", "serve")
		ollamaCmd.Start()
		time.Sleep(1 * time.Second)
	}

	ta := textarea.New()
	ta.Placeholder = "Send a message..."
	ta.Focus()
	ta.Prompt = "┃ "
	ta.CharLimit = 2000
	ta.SetHeight(3)
	ta.ShowLineNumbers = false

	pi := textarea.New()
	pi.Placeholder = "Enter a system prompt..."
	pi.Prompt = "┃ "
	pi.CharLimit = 2000
	pi.SetHeight(5)
	pi.ShowLineNumbers = false

	fi := textarea.New()
	fi.Placeholder = "/path/to/your/file.pdf"
	fi.Prompt = "┃ "
	fi.CharLimit = 500
	fi.SetHeight(1)
	fi.ShowLineNumbers = false // THIS FIXES THE WEIRD "1" IN THE BOX

	vp := viewport.New(80, 20)

	m := model{
		state:        stateLoading,
		textarea:     ta,
		promptInput:  pi,
		fileInput:    fi,
		viewport:     vp,
		focusMode:    "textarea",
		selectedMsg:  -1,
		autoScroll:   true,
		systemPrompt: cfg.SystemPrompt,
		config:       cfg,
		ollamaCmd:    ollamaCmd,
		swapIndex:    -1,
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v", err)
		os.Exit(1)
	}
}
