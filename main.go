package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"charm.land/glamour/v2"
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
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	System  string                 `json:"system,omitempty"`
	Images  []string               `json:"images,omitempty"`
	Stream  bool                   `json:"stream"`
	Think   *bool                  `json:"think,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
}
type OllamaGenerateRes struct {
	Response string `json:"response"`
	Thinking string `json:"thinking,omitempty"`
}

// --- Config & App Models ---
type Config struct {
	SystemPrompt string `json:"system_prompt"`
}

type ChatMessage struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
}

type Attachment struct {
	Name    string
	Path    string
	Text    string
	IsImage bool
	Base64  string
}

type state int

const (
	stateLoading state = iota
	stateSelectModel
	stateHostInput
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
	textareaStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(colorGray).Padding(0, 2)
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
	statusIsError    bool
	systemPrompt     string
	config           Config
	viewport         viewport.Model
	textarea         textarea.Model
	promptInput      textarea.Model
	fileInput        textarea.Model
	hostInput        textarea.Model
	attachments      []Attachment
	attachCursor     int
	swapIndex        int
	host             string

	// Capability flags
	supportsVision   bool
	supportsThinking bool
	supportsTools    bool
	useThinking      bool
	useTools         bool
	showThinking     bool

	// Glamour renderer
	renderer *glamour.TermRenderer

	ollamaCmd        *exec.Cmd
	errText          string
	keepServer       bool
	err              error
	streamChan       chan OllamaGenerateRes
	cancelStream     context.CancelFunc
	isThinking       bool
	autoScroll       bool
	isSwitchingModel bool
	historyFiles     []string
	historyCursor    int
	width            int
	height           int

	// Optimization fields
	liveBuilder         *strings.Builder
	liveThinkingBuilder *strings.Builder
	renderedCache       []string
	cacheWidth          int
	cacheSelected       int
	cacheFocusMode      string
	cacheShowThinking   bool
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
type modelCapabilitiesMsg struct {
	vision   bool
	thinking bool
	tools    bool
}
type streamChunkMsg struct {
	Response string
	Thinking string
}
type streamDoneMsg struct{}

// Init is the missing piece that satisfies the tea.Model interface
func (m model) Init() tea.Cmd {
	return fetchModels(m.host)
}

// fetchModels is a dynamic command to fetch models from a specific host
func fetchModels(host string) tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(host + "/api/tags")
		if err != nil {
			return errMsg{fmt.Errorf("failed to connect to %s: %v", host, err)}
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

func fetchCapabilities(modelName string) tea.Cmd {
	return func() tea.Msg {
		name := strings.ToLower(modelName)
		vision := strings.Contains(name, "vision") || strings.Contains(name, "llava") || strings.Contains(name, "bakllava")
		thinking := strings.Contains(name, "deepseek-r1") || strings.Contains(name, "qwen") || strings.Contains(name, "marco-o1")
		tools := strings.Contains(name, "llama3.1") || strings.Contains(name, "llama3.2") || strings.Contains(name, "qwen")
		return modelCapabilitiesMsg{vision: vision, thinking: thinking, tools: tools}
	}
}

func waitForStream(ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		res, ok := <-ch
		if !ok {
			return streamDoneMsg{}
		}
		return streamChunkMsg{Response: res.Response, Thinking: res.Thinking}
	}
}

func startGenerate(ctx context.Context, m model, finalPrompt string, ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		var imageB64 []string
		var textContext strings.Builder
		if len(m.attachments) > 0 {
			textContext.WriteString("Context from attached files:\n")
			for _, a := range m.attachments {
				if a.IsImage {
					imageB64 = append(imageB64, a.Base64)
				} else {
					textContext.WriteString(fmt.Sprintf("--- %s ---\n%s\n\n", a.Name, a.Text))
				}
			}
		}

		prompt := finalPrompt
		if len(m.attachments) > 0 && textContext.Len() > 0 {
			prompt = textContext.String() + "User Prompt: " + finalPrompt
		}

		options := make(map[string]interface{})
		if m.useTools {
			options["tools"] = true
		}

		var thinkPtr *bool
		if m.supportsThinking {
			thinkPtr = &m.useThinking
		}

		reqBody, _ := json.Marshal(OllamaGenerateReq{
			Model:   m.models[m.cursor],
			Prompt:  prompt,
			System:  m.systemPrompt,
			Images:  imageB64,
			Stream:  true,
			Think:   thinkPtr,
			Options: options,
		})

		req, err := http.NewRequestWithContext(ctx, "POST", m.host+"/api/generate", bytes.NewBuffer(reqBody))
		if err != nil {
			close(ch)
			return nil
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			close(ch)
			return nil
		}
		defer resp.Body.Close()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			var res OllamaGenerateRes
			json.Unmarshal(scanner.Bytes(), &res)
			if res.Response != "" || res.Thinking != "" {
				ch <- res
			}
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
	if err != nil {
		return "", err
	}
	defer f.Close()
	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", err
	}
	buf.ReadFrom(b)
	return buf.String(), nil
}

func extractXMLFromZip(path, targetFile string) (string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer r.Close()
	for _, f := range r.File {
		if f.Name == targetFile {
			rc, err := f.Open()
			if err != nil {
				return "", err
			}
			defer rc.Close()
			data, err := io.ReadAll(rc)
			return string(data), err
		}
	}
	return "", fmt.Errorf("could not find %s", targetFile)
}

func stripXML(xmlContent string) string {
	decoder := xml.NewDecoder(strings.NewReader(xmlContent))
	var textBuilder strings.Builder
	for {
		token, err := decoder.Token()
		if err != nil {
			break
		}
		if t, ok := token.(xml.CharData); ok {
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
	path = filepath.Clean(path)
	ext := strings.ToLower(filepath.Ext(path))

	var content string
	var err error
	isImage := false
	base64Str := ""

	switch ext {
		case ".txt", ".md", ".go", ".c", ".h", ".py", ".js", ".json", ".sh", ".yaml", ".yml", ".rs", ".toml", ".ini", ".conf":
			content, err = readTextFile(path)
		case ".pdf":
			content, err = readPDF(path)
		case ".docx":
			var xmlStr string
			xmlStr, err = extractXMLFromZip(path, "word/document.xml")
			if err == nil {
				content = stripXML(xmlStr)
			}
		case ".odt":
			var xmlStr string
			xmlStr, err = extractXMLFromZip(path, "content.xml")
			if err == nil {
				content = stripXML(xmlStr)
			}
		case ".abw":
			var xmlStr string
			xmlStr, err = readTextFile(path)
			if err == nil {
				content = stripXML(xmlStr)
			}
		case ".png", ".jpg", ".jpeg", ".webp":
			isImage = true
			data, errRead := os.ReadFile(path)
			if errRead == nil {
				base64Str = base64.StdEncoding.EncodeToString(data)
			}
			err = errRead
		default:
			return fileErrorMsg{err: fmt.Sprintf("unsupported file type: %s", ext)}
	}

	if err != nil {
		return fileErrorMsg{err: err.Error()}
	}

	if !isImage {
		content = strings.Join(strings.Fields(content), " ")
		if len(content) > 20000 {
			content = content[:20000] + "\n...[TRUNCATED]..."
		}
	} else {
		content = "[Image Data]"
	}

	return fileAttachedMsg{attachment: Attachment{Name: filepath.Base(path), Path: path, Text: content, IsImage: isImage, Base64: base64Str}}
}

func editInNano(path string, index int) tea.Cmd {
	c := exec.Command("nano", path)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		if err != nil {
			return fileErrorMsg{err: err.Error()}
		}
		content, err := readTextFile(path)
		if err != nil {
			return fileErrorMsg{err: err.Error()}
		}
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
			m.viewport.Height = msg.Height - 12
			m.textarea.SetWidth(msg.Width - 8)
			m.promptInput.SetWidth(msg.Width - 14)
			m.fileInput.SetWidth(msg.Width - 14)
			m.hostInput.SetWidth(msg.Width - 14)

			// Recreate Glamour renderer with new width
			r, _ := glamour.NewTermRenderer(
				glamour.WithStylePath("dark"),
							glamour.WithWordWrap(msg.Width-10),
			)
			m.renderer = r

			m.updateViewport()

		case clearStatusMsg:
			m.statusMsg = ""
			m.statusIsError = false

		case ollamaMissingMsg:
			m.state = stateError
			m.errText = "you need to install Ollama CLI tools\n\nGet it at: https://ollama.com/"
			m.keepServer = false

		case noModelsMsg:
			m.state = stateError
			m.errText = "please install a model with 'ollama pull <model name>'\n\nwe'll keep the server running when you exit\nuntil you install a model and relaunch."
			m.keepServer = true

		case modelCapabilitiesMsg:
			m.supportsVision = msg.vision
			m.supportsThinking = msg.thinking
			m.supportsTools = msg.tools
			m.useThinking = msg.thinking
			m.useTools = msg.tools
			m.updateViewport()

		case fileAttachedMsg:
			if m.swapIndex == -1 {
				m.attachments = append(m.attachments, msg.attachment)
			} else {
				m.attachments[m.swapIndex] = msg.attachment
				m.swapIndex = -1
			}
			m.state = stateAttachmentList
			m.statusMsg = fmt.Sprintf("Attached: %s", msg.attachment.Name)
			cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })

		case fileEditedMsg:
			m.attachments[msg.index].Text = msg.text
			m.state = stateAttachmentList
			m.statusMsg = "File edited and reloaded."
			cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })

		case fileErrorMsg:
			m.state = stateChat
			m.focusMode = "textarea"
			m.textarea.Focus()
			m.statusMsg = "File error: " + msg.err
			m.statusIsError = true
			cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })

		case modelsFetchedMsg:
			m.models = msg.models
			m.state = stateSelectModel

		case historyFilesMsg:
			m.historyFiles = msg.files
			m.historyCursor = 0
			m.state = stateHistory

		case streamChunkMsg:
			if msg.Thinking != "" {
				m.liveThinkingBuilder.WriteString(msg.Thinking)
				m.messages[len(m.messages)-1].Thinking = m.liveThinkingBuilder.String()
			}
			if msg.Response != "" {
				m.liveBuilder.WriteString(msg.Response)
				m.messages[len(m.messages)-1].Content = m.liveBuilder.String()
			}
			m.updateViewport()
			if m.autoScroll {
				m.viewport.GotoBottom()
			}
			cmds = append(cmds, waitForStream(m.streamChan))

		case streamDoneMsg:
			m.isThinking = false

		case errMsg:
			// Instead of crashing, show the error and return to the previous state
			m.statusMsg = msg.err.Error()
			m.statusIsError = true
			m.state = stateSelectModel // Go back to model selection so they can fix the host
			cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
			return m, tea.Batch(cmds...)

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
				if m.state == stateChat || m.state == stateHistory || m.state == stateSystemPrompt || m.state == stateFileAttach || m.state == stateAttachmentList || m.state == stateHostInput {
					m.state = stateExiting
					m.textarea.Blur()
					m.promptInput.Blur()
					m.fileInput.Blur()
					m.hostInput.Blur()
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
				if msg.String() == "n" || msg.String() == "N" {
					return m, tea.Quit
				}
				return m, nil
			}

			if m.state == stateHostInput {
				var cmd tea.Cmd
				m.hostInput, cmd = m.hostInput.Update(msg)
				cmds = append(cmds, cmd)

				if msg.String() == "esc" {
					m.state = stateSelectModel
					m.hostInput.Blur()
				} else if msg.String() == "enter" {
					newHost := strings.TrimSpace(m.hostInput.Value())
					if newHost != "" {
						m.host = newHost
					}
					m.state = stateLoading
					m.hostInput.Blur()
					cmds = append(cmds, fetchModels(m.host))
				}
				return m, tea.Batch(cmds...)
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
					cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
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
						if m.attachCursor > 0 {
							m.attachCursor--
						}
					case "down", "j":
						if m.attachCursor < len(m.attachments)-1 {
							m.attachCursor++
						}
					case "X":
						if len(m.attachments) > 0 {
							m.attachments = append(m.attachments[:m.attachCursor], m.attachments[m.attachCursor+1:]...)
							if m.attachCursor >= len(m.attachments) && m.attachCursor > 0 {
								m.attachCursor--
							}
							m.statusMsg = "Attachment deleted."
							cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
						}
					case "S":
						if len(m.attachments) > 0 {
							m.swapIndex = m.attachCursor
							m.state = stateFileAttach
							m.fileInput.Focus()
						}
					case "N":
						if len(m.attachments) > 0 && !m.attachments[m.attachCursor].IsImage {
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
					cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
				} else if msg.String() == "ctrl+s" || msg.String() == "enter" {
					m.systemPrompt = m.promptInput.Value()
					m.config.SystemPrompt = m.systemPrompt
					saveConfig(m.config)
					m.state = stateChat
					m.focusMode = "list"
					m.promptInput.Blur()
					m.statusMsg = "System prompt saved."
					cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateHistory {
				switch msg.String() {
					case "esc":
						m.state = stateChat
						m.focusMode = "list"
					case "up", "k":
						if m.historyCursor > 0 {
							m.historyCursor--
						}
					case "down", "j":
						if m.historyCursor < len(m.historyFiles)-1 {
							m.historyCursor++
						}
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
								cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
							}
						}
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateSelectModel {
				switch msg.String() {
					case "up", "k":
						if m.cursor > 0 {
							m.cursor--
						}
					case "down", "j":
						if m.cursor < len(m.models)-1 {
							m.cursor++
						}
					case "ctrl+h":
						m.state = stateHostInput
						m.hostInput.SetValue(m.host)
						m.hostInput.Focus()
						return m, nil
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
						cmds = append(cmds, fetchCapabilities(m.models[m.cursor]))
				}
				return m, tea.Batch(cmds...)
			}

			if m.state == stateChat {
				if msg.String() == "alt+s" && m.isThinking {
					if m.cancelStream != nil {
						m.cancelStream()
					}
					m.isThinking = false
					m.messages[len(m.messages)-1].Content += " [Stopped by user]"
					m.updateViewport()
					return m, nil
				}

				if msg.String() == "tab" || msg.String() == "esc" {
					if m.focusMode == "textarea" {
						m.focusMode = "list"
						m.textarea.Blur()
						if m.selectedMsg == -1 {
							m.selectedMsg = len(m.messages) - 1
						}
					} else {
						m.focusMode = "textarea"
						m.textarea.Focus()
					}
					m.updateViewport()
					return m, nil
				}

				if msg.String() == "ctrl+t" {
					if m.supportsThinking {
						m.useThinking = !m.useThinking
						m.statusMsg = fmt.Sprintf("Thinking: %v", m.useThinking)
						cmds = append(cmds, func() tea.Msg { time.Sleep(3 * time.Second); return clearStatusMsg{} })
					} else {
						m.statusMsg = "Sorry, this model does not support Thinking."
						m.statusIsError = true
						cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
					}
					return m, tea.Batch(cmds...)
				}
				if msg.String() == "ctrl+y" {
					if m.supportsTools {
						m.useTools = !m.useTools
						m.statusMsg = fmt.Sprintf("Tools: %v", m.useTools)
						cmds = append(cmds, func() tea.Msg { time.Sleep(3 * time.Second); return clearStatusMsg{} })
					} else {
						m.statusMsg = "Sorry, this model does not support Tools."
						m.statusIsError = true
						cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
					}
					return m, tea.Batch(cmds...)
				}

				if msg.String() == "alt+t" {
					m.showThinking = !m.showThinking
					m.statusMsg = fmt.Sprintf("Show Thinking: %v", m.showThinking)
					cmds = append(cmds, func() tea.Msg { time.Sleep(3 * time.Second); return clearStatusMsg{} })
					m.updateViewport()
					return m, tea.Batch(cmds...)
				}

				if m.focusMode == "list" {
					switch msg.String() {
						case "up", "k":
							if m.selectedMsg > 0 {
								m.selectedMsg--
								m.autoScroll = false
								m.updateViewport()
							}
						case "down", "j":
							if m.selectedMsg < len(m.messages)-1 {
								m.selectedMsg++
								if m.selectedMsg == len(m.messages)-1 {
									m.autoScroll = true
								}
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
								cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
							}
						case "N":
							m.messages = []ChatMessage{{Role: "System", Content: fmt.Sprintf("Started new chat with %s", m.models[m.cursor])}}
							m.selectedMsg = 0
							m.autoScroll = true
							m.updateViewport()
							m.statusMsg = "Started new chat."
							cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
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
							cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
						case "B":
							if m.selectedMsg >= 0 && m.selectedMsg < len(m.messages) {
								m.messages = m.messages[:m.selectedMsg+1]
								m.messages = append(m.messages, ChatMessage{Role: "System", Content: "--- Branched from here ---"})
								m.selectedMsg = len(m.messages) - 1
								m.autoScroll = true
								m.updateViewport()
								m.statusMsg = "Branched chat."
								cmds = append(cmds, func() tea.Msg { time.Sleep(5 * time.Second); return clearStatusMsg{} })
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
						m.messages = append(m.messages, ChatMessage{Role: "You", Content: userPrompt})
						for _, a := range m.attachments {
							if a.IsImage {
								m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("🖼️ Image: %s", a.Name)})
							} else {
								m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("📎 Attached: %s", a.Name)})
							}
						}
						m.messages = append(m.messages, ChatMessage{Role: "AI", Content: ""})

						m.selectedMsg = -1
						m.streamChan = make(chan OllamaGenerateRes)
						m.isThinking = true
						m.autoScroll = true

						ctx, cancel := context.WithCancel(context.Background())
						m.cancelStream = cancel

						attachmentsCopy := make([]Attachment, len(m.attachments))
						copy(attachmentsCopy, m.attachments)

						m.textarea.Reset()
						m.liveBuilder.Reset()
						m.liveThinkingBuilder.Reset()
						m.updateViewport()
						m.viewport.GotoBottom()

						mCopy := m
						mCopy.attachments = attachmentsCopy

						cmds = append(cmds, startGenerate(ctx, mCopy, userPrompt, m.streamChan))
						cmds = append(cmds, waitForStream(m.streamChan))

						m.attachments = nil
					}
				}
			}
	}

	return m, tea.Batch(cmds...)
}

// Optimized updateViewport with caching
func (m *model) updateViewport() {
	if m.cacheWidth != m.viewport.Width || m.cacheSelected != m.selectedMsg || m.cacheFocusMode != m.focusMode || m.cacheShowThinking != m.showThinking || len(m.renderedCache) != len(m.messages) {
		m.cacheWidth = m.viewport.Width
		m.cacheSelected = m.selectedMsg
		m.cacheFocusMode = m.focusMode
		m.cacheShowThinking = m.showThinking
		m.renderedCache = make([]string, len(m.messages))
	}

	textWidth := m.viewport.Width - 3
	var sb strings.Builder
	selectedMsgStartLine := 0
	selectedMsgEndLine := 0
	currentLine := 0

	for i, msg := range m.messages {
		if len(m.renderedCache[i]) == 0 || i == len(m.messages)-1 {
			m.renderedCache[i] = m.renderMessage(msg, i, textWidth)
		}

		if i == m.selectedMsg {
			selectedMsgStartLine = currentLine
			lines := strings.Count(m.renderedCache[i], "\n") + 1
			selectedMsgEndLine = currentLine + lines - 1
			currentLine += lines
		} else {
			currentLine += strings.Count(m.renderedCache[i], "\n") + 1
		}

		sb.WriteString(m.renderedCache[i])
		sb.WriteString("\n")
	}

	m.viewport.SetContent(sb.String())

	if m.focusMode == "list" && m.selectedMsg >= 0 {
		if selectedMsgStartLine < m.viewport.YOffset {
			m.viewport.SetYOffset(selectedMsgStartLine)
		} else if selectedMsgEndLine >= m.viewport.YOffset+m.viewport.Height {
			m.viewport.SetYOffset(selectedMsgEndLine - m.viewport.Height + 1)
		}
	} else if m.autoScroll {
		m.viewport.GotoBottom()
	}
}

func (m *model) renderMessage(msg ChatMessage, i int, textWidth int) string {
	isSelected := (i == m.selectedMsg && m.focusMode == "list")

	var label string
	if msg.Role == "You" {
		label = "You:"
	} else if msg.Role == "AI" {
		label = "AI:"
	} else {
		label = "•"
	}

	pointer := "   "
	if isSelected {
		pointer = lipgloss.NewStyle().Foreground(colorCyan).Bold(true).Render(" > ")
	}

	var styledLabel string
	if isSelected {
		bg := colorCyan
		if msg.Role == "AI" {
			bg = colorPurple
		}
		if msg.Role == "System" {
			bg = colorGray
		}
		styledLabel = lipgloss.NewStyle().Background(bg).Foreground(lipgloss.Color("0")).Bold(true).Render(" " + label + " ")
	} else {
		if msg.Role == "You" {
			styledLabel = userLabelStyle.Render(label)
		} else if msg.Role == "AI" {
			styledLabel = aiLabelStyle.Render(label)
		} else {
			styledLabel = lipgloss.NewStyle().Foreground(colorGray).Italic(true).Render(label)
		}
	}

	labelWidth := lipgloss.Width(styledLabel)
	wrapWidth := textWidth - labelWidth - 1
	if wrapWidth < 10 {
		wrapWidth = 10
	}

	var finalContent strings.Builder

	if msg.Role == "AI" && msg.Thinking != "" && m.showThinking {
		thoughtStyle := lipgloss.NewStyle().Foreground(colorGray).Italic(true)
		wrappedThoughts := lipgloss.NewStyle().Width(wrapWidth).Render(msg.Thinking)
		finalContent.WriteString(thoughtStyle.Render("🤔 [Thinking...]"))
		finalContent.WriteString("\n")
		finalContent.WriteString(thoughtStyle.Render(wrappedThoughts))
		finalContent.WriteString("\n\n")
	}

	var renderedText string
	if msg.Role == "AI" && m.renderer != nil {
		out, err := m.renderer.Render(msg.Content)
		if err == nil {
			renderedText = strings.TrimSuffix(out, "\n")
		} else {
			renderedText = lipgloss.NewStyle().Width(wrapWidth).Render(msg.Content)
		}
	} else {
		renderedText = lipgloss.NewStyle().Width(wrapWidth).Render(msg.Content)
	}
	finalContent.WriteString(renderedText)

	lines := strings.Split(finalContent.String(), "\n")

	var combined []string
	combined = append(combined, styledLabel+" "+lines[0])
	indent := strings.Repeat(" ", labelWidth+1)
	for j := 1; j < len(lines); j++ {
		combined = append(combined, indent+lines[j])
	}

	block := strings.Join(combined, "\n")
	return lipgloss.JoinHorizontal(lipgloss.Top, pointer, block)
}

func (m model) renderScrollableList(items []string, cursor int, maxHeight int) string {
	if maxHeight < 5 {
		maxHeight = 5
	}
	start := 0
	if cursor >= maxHeight {
		start = cursor - maxHeight + 1
	}
	end := start + maxHeight
	if end > len(items) {
		end = len(items)
	}

	var s strings.Builder
	if start > 0 {
		s.WriteString(fmt.Sprintf("  ↑ (%d more)\n", start))
	}
	for i := start; i < end; i++ {
		c := " "
		style := lipgloss.NewStyle()
		if i == cursor {
			c = ">"
			style = style.Foreground(colorCyan).Bold(true)
		}
		s.WriteString(style.Render(fmt.Sprintf("%s %s", c, items[i])) + "\n")
	}
	if end < len(items) {
		s.WriteString(fmt.Sprintf("  ↓ (%d more)\n", len(items)-end))
	}
	return s.String()
}

func (m model) View() string {
	if m.err != nil {
		return fmt.Sprintf("Error: %v\n", m.err)
	}

	if m.state == stateError {
		errBox := lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(colorRed).Padding(1, 2).
		Render(lipgloss.NewStyle().Foreground(colorRed).Bold(true).Render("ERROR: ") + m.errText)
		return lipgloss.NewStyle().Margin(2, 4).Render(errBox)
	}

	if m.state == stateLoading {
		return "\n  Connecting to Ollama and fetching models...\n"
	}
	if m.state == stateExiting {
		return "\n  Do you want to stop the Ollama server? (y/n)\n"
	}

	if m.state == stateHostInput {
		box := textareaStyle.Width(m.width - 10).Render(m.hostInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Press Enter to save and connect, Esc to cancel.")
		content := lipgloss.JoinVertical(lipgloss.Left, aiLabelStyle.Render("Host URL (e.g. http://192.168.1.100:11434)"), "", box, "", help)
		return lipgloss.NewStyle().Margin(1, 2).Render(content)
	}

	if m.state == stateFileAttach {
		title := "Attach File (Ctrl+F)"
		if m.swapIndex != -1 {
			title = "Replace Attachment (Shift+S)"
		}
		box := textareaStyle.Width(m.width - 10).Render(m.fileInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Drag & drop a file or type a path. Press Enter to attach, Esc to cancel.")
		content := lipgloss.JoinVertical(lipgloss.Left, aiLabelStyle.Render(title), "", box, "", help)
		return lipgloss.NewStyle().Margin(1, 2).Render(content)
	}

	if m.state == stateAttachmentList {
		var items []string
		for _, a := range m.attachments {
			icon := "📎"
			if a.IsImage {
				icon = "🖼️"
			}
			items = append(items, fmt.Sprintf("%s %s (%d chars)", icon, a.Name, len(a.Text)))
		}
		title := "Manage Attachments (Ctrl+L)\n\n"
		if len(m.attachments) == 0 {
			title += "No files attached.\nPress Esc to return."
		} else {
			title += m.renderScrollableList(items, m.attachCursor, m.height-8)
			title += "\n↑/↓ Select • Shift+X Delete • Shift+S Replace • Shift+N Edit in Nano • Esc/Ctrl+S Save & Return"
		}
		return lipgloss.NewStyle().Margin(1, 2).Render(title)
	}

	if m.state == stateSystemPrompt {
		box := textareaStyle.Width(m.width - 10).Render(m.promptInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Press Enter or Ctrl+S to save, Esc to cancel.")
		content := lipgloss.JoinVertical(lipgloss.Left, aiLabelStyle.Render("Edit System Prompt"), "", box, "", help)
		return lipgloss.NewStyle().Margin(1, 2).Render(content)
	}

	if m.state == stateHistory {
		var items []string
		for _, f := range m.historyFiles {
			items = append(items, filepath.Base(f))
		}
		title := "Select a chat history to load (Up/Down, Enter to load, Esc to cancel):\n\n"
		if len(items) == 0 {
			title += "No saved histories found.\n"
		} else {
			title += m.renderScrollableList(items, m.historyCursor, m.height-8)
		}
		return lipgloss.NewStyle().Margin(1, 2).Render(title)
	}

	if m.state == stateSelectModel {
		title := fmt.Sprintf("Select a model (Host: %s)\nPress Ctrl+H to change host.\n\n", m.host)
		title += m.renderScrollableList(m.models, m.cursor, m.height-8)
		return lipgloss.NewStyle().Margin(1, 2).Render(title)
	}

	// --- CHAT STATE UI ---
	modelName := m.models[m.cursor]
	totalChars := 0
	for _, msg := range m.messages {
		totalChars += len(msg.Content) + len(msg.Thinking)
	}
	for _, a := range m.attachments {
		totalChars += len(a.Text)
	}

	approxTokens := totalChars / 4
	percent := (float64(approxTokens) / float64(4096)) * 100
	meterColor := colorGreen
	if percent > 75 {
		meterColor = colorYellow
	}
	if percent > 90 {
		meterColor = colorRed
	}
	meter := lipgloss.NewStyle().Foreground(meterColor).Render(fmt.Sprintf("[%.0f%% Context]", percent))

	capStr := ""
	if m.supportsVision {
		capStr += " [👁️ Vision]"
	}
	if m.supportsThinking {
		status := "OFF"
		if m.useThinking {
			status = "ON"
		}
		capStr += fmt.Sprintf(" [🧠 Thinking: %s]", status)
	}
	if m.supportsTools {
		status := "OFF"
		if m.useTools {
			status = "ON"
		}
		capStr += fmt.Sprintf(" [🛠️ Tools: %s]", status)
	}

	headerText := fmt.Sprintf("Model: %s %s %s", aiLabelStyle.Render(modelName), meter, lipgloss.NewStyle().Foreground(colorGray).Render(capStr))
	header := headerStyle.Width(m.width - 2).Render(headerText)

	vpView := m.viewport.View()
	taView := textareaStyle.Render(m.textarea.View())

	var statusText string
	if m.statusMsg != "" {
		if m.statusIsError {
			statusText = lipgloss.NewStyle().Foreground(colorRed).Render(m.statusMsg)
		} else {
			statusText = lipgloss.NewStyle().Foreground(colorGreen).Render(m.statusMsg)
		}
	} else if m.isThinking {
		statusText = lipgloss.NewStyle().Foreground(colorRed).Render("Thinking... (Alt+S to stop)")
	} else if m.focusMode == "list" {
		listHelp := "[LIST] w/z/s Scroll • ↑/↓ Select • C Copy • N New • S Switch • P Prompt • J Save • B Branch • Ctrl+O Hist"
		if m.supportsThinking {
			listHelp += " • Ctrl+T Think"
		}
		if m.supportsTools {
			listHelp += " • Ctrl+Y Tools"
		}
		statusText = lipgloss.NewStyle().Foreground(colorPurple).Render(listHelp)
	} else {
		chatHelp := "[CHAT MODE] Ctrl+F to attach file • Tab for list"
		if len(m.attachments) > 0 {
			chatHelp = lipgloss.NewStyle().Foreground(colorYellow).Bold(true).Render(fmt.Sprintf("[%d Attached] ", len(m.attachments))) + "[CHAT MODE] Ctrl+L to manage • Tab for list"
		}
		if m.supportsThinking {
			chatHelp += " • Ctrl+T Think • Alt+T View Thoughts"
		}
		if m.supportsTools {
			chatHelp += " • Ctrl+Y for Tooling mode"
		}
		statusText = chatHelp
	}
	status := statusStyle.Width(m.width - 2).Render(statusText)

	return lipgloss.JoinVertical(lipgloss.Left, header, vpView, taView, status)
}

func main() {
	hostFlag := flag.String("host", "http://localhost:11434", "Ollama server URL")
	flag.Parse()

	cfg := loadConfig()

	if *hostFlag == "http://localhost:11434" {
		if _, err := exec.LookPath("ollama"); err == nil {
			cmd := exec.Command("ollama", "serve")
			cmd.Start()
			time.Sleep(1 * time.Second)
		}
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
	fi.Placeholder = "/path/to/your/file.pdf or image.png"
	fi.Prompt = "┃ "
	fi.CharLimit = 500
	fi.SetHeight(1)
	fi.ShowLineNumbers = false

	hi := textarea.New()
	hi.Placeholder = "http://localhost:11434"
	hi.Prompt = "┃ "
	hi.CharLimit = 500
	hi.SetHeight(1)
	hi.ShowLineNumbers = false

	vp := viewport.New(80, 20)

	r, _ := glamour.NewTermRenderer(
		glamour.WithStylePath("dark"),
					glamour.WithWordWrap(70),
	)

	m := model{
		state:               stateLoading,
		textarea:            ta,
		promptInput:         pi,
		fileInput:           fi,
		hostInput:           hi,
		host:                *hostFlag,
		viewport:            vp,
		focusMode:           "textarea",
		selectedMsg:         -1,
		autoScroll:          true,
		systemPrompt:        cfg.SystemPrompt,
		config:              cfg,
		swapIndex:           -1,
		liveBuilder:         &strings.Builder{},
		liveThinkingBuilder: &strings.Builder{},
		showThinking:        true,
		renderer:            r,
	}

	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
