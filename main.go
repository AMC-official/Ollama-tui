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
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ledongthuc/pdf"
)

// --- Constants & shared helpers ---
const (
	defaultHost         = "http://localhost:11434"
	contextWindowTokens = 4096 // Ollama's default num_ctx; only used for the context meter
	maxAttachChars      = 20000
	maxToolRounds       = 5 // how many rounds of tool calls one answer may take
)

// apiClient is for the short, non-streaming calls (tags / show) so a dead host
// can't hang the UI forever. Streaming chat builds its own request.
var apiClient = &http.Client{Timeout: 15 * time.Second}

// File extensions that are read as plain text (and can be edited in nano).
var textExts = map[string]bool{
	".txt": true, ".md": true, ".go": true, ".c": true, ".h": true, ".py": true,
	".js": true, ".json": true, ".sh": true, ".yaml": true, ".yml": true,
	".rs": true, ".toml": true, ".ini": true, ".conf": true,
}

// --- Ollama API Structs ---
type OllamaModel struct {
	Name string `json:"name"`
}
type OllamaTags struct {
	Models []OllamaModel `json:"models"`
}

// FIXED: we now talk to /api/chat (which takes the whole conversation) instead
// of /api/generate (which only ever saw the latest prompt).
type OllamaChatMsg struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Images     []string   `json:"images,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"` // set on an assistant message that asks for tools
	ToolName   string     `json:"tool_name,omitempty"`  // set on a "tool" message: which tool answered
	ToolCallID string     `json:"tool_call_id,omitempty"`
}
type OllamaChatReq struct {
	Model    string                 `json:"model"`
	Messages []OllamaChatMsg        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Think    *bool                  `json:"think,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
	Tools    []ToolDef              `json:"tools,omitempty"`
}

// One streamed line of an /api/chat response.
type OllamaChatRes struct {
	Message struct {
		Content   string     `json:"content"`
		Thinking  string     `json:"thinking,omitempty"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Error           string `json:"error,omitempty"`
	Done            bool   `json:"done"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
	EvalDuration    int64  `json:"eval_duration,omitempty"` // nanoseconds
}

// What the streaming goroutine hands over to the UI.
// UsageInfo is what a server reports about the tokens of one reply.
type UsageInfo struct {
	Prompt       int
	Completion   int
	TokensPerSec float64
}

type OllamaGenerateRes struct {
	Response  string
	Thinking  string
	Error     string
	ToolCalls []ToolCall
	Usage     *UsageInfo
}

// --- Tool calling ---

// ToolCall is the model asking us to run one of the tools.
type ToolCallFunction struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}
type ToolCall struct {
	ID       string           `json:"id,omitempty"` // set by cloud providers; needed to match results to calls
	Function ToolCallFunction `json:"function"`
}

// ToolDef describes one tool to the model (name, what it does, its parameters).
type ToolDefFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}
type ToolDef struct {
	Type     string          `json:"type"`
	Function ToolDefFunction `json:"function"`
}
type OllamaShowRes struct {
	Capabilities []string               `json:"capabilities"`
	ModelInfo    map[string]interface{} `json:"model_info"`
}

// --- Config & App Models ---
type Config struct {
	SystemPrompt  string   `json:"system_prompt"`
	Temperature   *float64 `json:"temperature,omitempty"` // nil = model default
	TopP          *float64 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	NumCtx        int      `json:"num_ctx,omitempty"`        // 0 = Ollama default
	Backend       string   `json:"backend,omitempty"`        // "ollama" (default), "openrouter" or "openai"
	OpenRouterKey string   `json:"openrouter_key,omitempty"` // plain text; the config file is only readable by you
	OpenAIKey     string   `json:"openai_key,omitempty"`
}

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Thinking   string     `json:"thinking,omitempty"`
	Context    string     `json:"context,omitempty"`    // text of files attached to this user message
	Elapsed    string     `json:"elapsed,omitempty"`    // how long an AI reply took (shown under it)
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"` // tools the AI asked for in this message
	ToolName   string     `json:"tool_name,omitempty"`  // for "Tool" messages: which tool produced the result
	ToolCallID string     `json:"tool_call_id,omitempty"`
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
	stateSettings
	stateFileAttach
	stateAttachmentList
	stateModelPull
	stateAPIKeys
	stateError
	stateExiting
)

// The steps of the "pull a model" screen.
type pullPhase int

const (
	pullInput       pullPhase = iota // typing the model name and tag
	pullChecking                     // asking the Ollama library whether it exists
	pullConfirm                      // found: waiting for y/n
	pullDownloading                  // downloading, showing the progress bar
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
	state         state
	models        []string
	cursor        int
	messages      []ChatMessage
	selectedMsg   int
	focusMode     string
	statusMsg     string
	statusIsError bool
	systemPrompt  string
	config        Config
	viewport      viewport.Model
	textarea      textarea.Model
	promptInput   textarea.Model
	settingInputs [4]textarea.Model // temperature, top p, top k, context tokens
	settingFocus  int               // 0 = system prompt, 1-4 = the inputs above
	fileInput     textarea.Model
	hostInput     textarea.Model
	attachments   []Attachment
	attachCursor  int
	swapIndex     int
	host          string

	// Capability flags
	supportsVision   bool
	supportsThinking bool
	supportsTools    bool
	useThinking      bool
	useTools         bool
	modelMaxCtx      int // context tokens the current model supports (0 = unknown)
	showThinking     bool

	// Glamour renderer
	renderer *glamour.TermRenderer

	ollamaCmd        *exec.Cmd
	ollamaMissing    bool
	errText          string
	keepServer       bool
	err              error
	streamChan       chan OllamaGenerateRes
	cancelStream     context.CancelFunc
	isThinking       bool
	genStart         time.Time  // when the current reply was requested
	gotOutput        bool       // true once the model has sent any text
	pendingCalls     []ToolCall // tools the model asked for during the current reply
	toolRounds       int        // tool rounds used by the current answer
	autoScroll       bool
	isSwitchingModel bool
	deleteTarget     string            // model waiting for a y/n delete confirmation
	selectAfterFetch string            // highlight this model after the next model-list refresh
	pullInputs       [2]textarea.Model // model name, parameters/tag
	pullFocus        int
	pullStep         pullPhase
	pullModel        string // name:tag being pulled
	pullSize         int64  // total download size from the library check
	pullTotal        int64
	pullDone         int64
	pullStatus       string
	pullErr          string
	pullStart        time.Time
	pullChan         chan pullEvent
	pullCancel       context.CancelFunc
	pullSuccess      bool
	backend          string                // backendOllama, backendOpenRouter or backendOpenAI
	cloudInfo        map[string]cloudModel // what we know about the models in the cloud list
	filter           string                // text the model list is filtered by
	filterMode       bool                  // typing the filter right now?
	filterInput      textinput.Model
	keyInputs        [2]textinput.Model // OpenRouter key, OpenAI key (masked)
	keyFocus         int
	listError        string // why the model list is empty, if loading it failed
	dirty            bool   // streamed text waits for the next render tick
	snappedMsg       int    // the selected message we last scrolled into view
	console          bool   // the console overlay is open
	consoleVP        viewport.Model
	consoleIn        textinput.Model
	consoleLines     []string
	consoleHist      []string
	consoleHistPos   int
	konami           int // progress of a key sequence on the settings screen
	usagePrompt      int // token counts reported by the servers, this run
	usageCompletion  int
	usageReplies     int
	lastUsage        UsageInfo
	ccChan           chan OllamaGenerateRes // a one-off :chat prompt from the console
	ccCancel         context.CancelFunc
	ccPrompt         string
	ccText           string
	ccThinking       string
	ccStart          time.Time
	ccFirst          time.Time
	ccUsage          *UsageInfo
	showStats        bool // CPU / RAM / GPU percentages in the header (Alt+H)
	statsGen         int  // identifies the running stats ticker
	haveStats        bool
	stats            SystemStats
	cpuPrev          cpuTimes
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
	os.MkdirAll(filepath.Dir(path), 0700)
	data, _ := json.MarshalIndent(cfg, "", "  ")
	// The file can hold API keys, so only you may read it.
	os.WriteFile(path, data, 0600)
	os.Chmod(path, 0600)
}

// --- Misc Helpers ---

// FIXED: one place that decides where chat histories live (used for both
// saving and loading, and works where $HOME isn't set, e.g. Windows).
// dataDir is the folder for everything Ollama-TUI saves.
func dataDir() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = os.Getenv("HOME")
	}
	return filepath.Join(home, ".local", "share", "ollama-tui")
}

// historyDir holds the saved chats, consoleLogDir the saved console logs.
func historyDir() string    { return filepath.Join(dataDir(), "chats") }
func consoleLogDir() string { return filepath.Join(dataDir(), "console") }

// migrateLegacyChats moves chats saved by older versions (straight in the data
// folder) into the chats folder.
func migrateLegacyChats() {
	old, _ := filepath.Glob(filepath.Join(dataDir(), "chat_*.json"))
	if len(old) == 0 {
		return
	}
	if err := os.MkdirAll(historyDir(), 0755); err != nil {
		return
	}
	for _, f := range old {
		os.Rename(f, filepath.Join(historyDir(), filepath.Base(f)))
	}
}

func normalizeHost(h string) string {
	h = strings.TrimRight(strings.TrimSpace(h), "/")
	if h != "" && !strings.Contains(h, "://") {
		h = "http://" + h
	}
	return h
}

// stopOllama shuts down the server we started. FIXED: it used to be force-killed
// (SIGKILL), which doesn't give Ollama a chance to stop its llama-server runner
// processes, so those were left running and kept the model in RAM. Now we ask
// politely first and only force-kill if it doesn't exit in time.
func stopOllama(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		// e.g. Windows can't send an interrupt: fall back to a hard kill.
		cmd.Process.Kill()
		return
	}
	done := make(chan struct{})
	go func() {
		cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		cmd.Process.Kill()
	}
}
func ollamaReachable(host string) bool {
	c := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := c.Get(host + "/api/tags")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return true
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
type fileEditedMsg struct {
	index int
	text  string
}
type modelCapabilitiesMsg struct {
	vision   bool
	thinking bool
	tools    bool
	maxCtx   int // 0 = unknown
}
type streamChunkMsg struct {
	ch        chan OllamaGenerateRes // which stream this chunk belongs to
	Response  string
	Thinking  string
	Error     string
	ToolCalls []ToolCall
	Usage     *UsageInfo
}
type streamDoneMsg struct{ ch chan OllamaGenerateRes }

// tickMsg arrives once a second while a reply is being generated.
type tickMsg struct{ ch chan OllamaGenerateRes }

// pullEvent is one progress update from a model download.
type pullEvent struct {
	Status    string
	Completed int64
	Total     int64
	Err       string
}
type pullProgressMsg struct {
	ch chan pullEvent
	ev pullEvent
}
type pullDoneMsg struct{ ch chan pullEvent }

// pullCheckedMsg is the answer of the library check done before a download.
type pullCheckedMsg struct {
	ref      string
	size     int64
	notFound bool
	err      error
}
type deleteDoneMsg struct {
	name string
	err  error
}

// Init is the missing piece that satisfies the tea.Model interface
func (m model) Init() tea.Cmd {
	if m.isCloud() {
		return fetchCloudModels(m.backend, m.apiKey())
	}
	if m.ollamaMissing {
		return func() tea.Msg { return ollamaMissingMsg{} }
	}
	return fetchModels(m.host)
}

// fetchModels is a dynamic command to fetch models from a specific host
func fetchModels(host string) tea.Cmd {
	return func() tea.Msg {
		resp, err := apiClient.Get(host + "/api/tags")
		if err != nil {
			return errMsg{fmt.Errorf("failed to connect to %s: %v", host, err)}
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return errMsg{fmt.Errorf("%s answered with %s", host, resp.Status)}
		}
		var tags OllamaTags
		if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
			return errMsg{fmt.Errorf("unexpected response from %s: %v", host, err)}
		}
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

func fetchCapabilities(host, modelName string) tea.Cmd {
	return func() tea.Msg {
		maxCtx := 0

		// Preferred: ask Ollama itself. Recent versions return a "capabilities"
		// list, and model_info says how many context tokens the model supports.
		body, _ := json.Marshal(map[string]string{"model": modelName, "name": modelName})
		if resp, err := apiClient.Post(host+"/api/show", "application/json", bytes.NewReader(body)); err == nil {
			defer resp.Body.Close()
			var show OllamaShowRes
			if resp.StatusCode == http.StatusOK && json.NewDecoder(resp.Body).Decode(&show) == nil {
				maxCtx = contextLengthFromInfo(show.ModelInfo)
				if len(show.Capabilities) > 0 {
					caps := modelCapabilitiesMsg{maxCtx: maxCtx}
					for _, c := range show.Capabilities {
						switch c {
						case "vision":
							caps.vision = true
						case "thinking":
							caps.thinking = true
						case "tools":
							caps.tools = true
						}
					}
					return caps
				}
			}
		}

		// Fallback for older servers: guess from the model name.
		name := strings.ToLower(modelName)
		vision := strings.Contains(name, "vision") || strings.Contains(name, "llava") || strings.Contains(name, "bakllava")
		thinking := strings.Contains(name, "deepseek-r1") || strings.Contains(name, "qwen") || strings.Contains(name, "marco-o1")
		tools := strings.Contains(name, "llama3.1") || strings.Contains(name, "llama3.2") || strings.Contains(name, "qwen")
		return modelCapabilitiesMsg{vision: vision, thinking: thinking, tools: tools, maxCtx: maxCtx}
	}
}

// contextLengthFromInfo reads "<architecture>.context_length" from the
// model_info object that /api/show returns. 0 means "unknown".
func contextLengthFromInfo(info map[string]interface{}) int {
	if arch, ok := info["general.architecture"].(string); ok {
		if v, ok := info[arch+".context_length"].(float64); ok {
			return int(v)
		}
	}
	for k, v := range info {
		if strings.HasSuffix(k, ".context_length") {
			if f, ok := v.(float64); ok {
				return int(f)
			}
		}
	}
	return 0
}
func waitForStream(ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		res, ok := <-ch
		if !ok {
			return streamDoneMsg{ch: ch}
		}
		return streamChunkMsg{ch: ch, Response: res.Response, Thinking: res.Thinking, Error: res.Error, ToolCalls: res.ToolCalls, Usage: res.Usage}
	}
}

// buildChatMessages turns the on-screen history into the message list Ollama
// expects. UI-only "System" notes are skipped; the system prompt goes first.
// Images are attached to the most recent user message only.
func buildChatMessages(history []ChatMessage, systemPrompt string, images []string) []OllamaChatMsg {
	var out []OllamaChatMsg
	if strings.TrimSpace(systemPrompt) != "" {
		out = append(out, OllamaChatMsg{Role: "system", Content: systemPrompt})
	}
	for _, h := range history {
		switch h.Role {
		case "You":
			content := h.Content
			if h.Context != "" {
				content = h.Context + "User Prompt: " + h.Content
			}
			out = append(out, OllamaChatMsg{Role: "user", Content: content})
		case "AI":
			content := strings.TrimSuffix(h.Content, " [Stopped by user]")
			if strings.TrimSpace(content) == "" && len(h.ToolCalls) == 0 {
				continue
			}
			out = append(out, OllamaChatMsg{Role: "assistant", Content: content, ToolCalls: h.ToolCalls})
		case "Tool":
			out = append(out, OllamaChatMsg{Role: "tool", ToolName: h.ToolName, ToolCallID: h.ToolCallID, Content: h.Content})
		}
	}
	if len(images) > 0 {
		for i := len(out) - 1; i >= 0; i-- {
			if out[i].Role == "user" {
				out[i].Images = images
				break
			}
		}
	}
	return out
}

// startGenerate streams a reply into ch and closes it when finished.
// FIXED: connection problems, HTTP errors and {"error": "..."} lines are now
// reported to the UI instead of silently producing an empty answer.
func startGenerate(ctx context.Context, host, modelName string, msgs []OllamaChatMsg, think *bool, options map[string]interface{}, tools []ToolDef, ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		defer close(ch)

		// send returns false if the stream was cancelled while waiting.
		send := func(res OllamaGenerateRes) bool {
			select {
			case ch <- res:
				return true
			case <-ctx.Done():
				return false
			}
		}

		reqBody, err := json.Marshal(OllamaChatReq{
			Model:    modelName,
			Messages: msgs,
			Stream:   true,
			Think:    think,
			Options:  options,
			Tools:    tools,
		})
		if err != nil {
			send(OllamaGenerateRes{Error: err.Error()})
			return nil
		}

		req, err := http.NewRequestWithContext(ctx, "POST", host+"/api/chat", bytes.NewBuffer(reqBody))
		if err != nil {
			send(OllamaGenerateRes{Error: err.Error()})
			return nil
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			if ctx.Err() == nil {
				send(OllamaGenerateRes{Error: fmt.Sprintf("failed to reach %s: %v", host, err)})
			}
			return nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			detail := strings.TrimSpace(string(body))
			var e struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(body, &e) == nil && e.Error != "" {
				detail = e.Error
			}
			send(OllamaGenerateRes{Error: fmt.Sprintf("Ollama answered %s: %s", resp.Status, detail)})
			return nil
		}

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			var line OllamaChatRes
			if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
				continue
			}
			if line.Error != "" {
				send(OllamaGenerateRes{Error: line.Error})
				return nil
			}
			if line.Message.Content != "" || line.Message.Thinking != "" || len(line.Message.ToolCalls) > 0 {
				if !send(OllamaGenerateRes{Response: line.Message.Content, Thinking: line.Message.Thinking, ToolCalls: line.Message.ToolCalls}) {
					return nil
				}
			}
			if line.Done && (line.PromptEvalCount > 0 || line.EvalCount > 0) {
				u := &UsageInfo{Prompt: line.PromptEvalCount, Completion: line.EvalCount}
				if line.EvalDuration > 0 {
					u.TokensPerSec = float64(line.EvalCount) / (float64(line.EvalDuration) / 1e9)
				}
				if !send(OllamaGenerateRes{Usage: u}) {
					return nil
				}
			}
		}
		if err := scanner.Err(); err != nil && ctx.Err() == nil {
			send(OllamaGenerateRes{Error: "connection lost: " + err.Error()})
		}
		return nil
	}
}

func loadHistoryFiles() tea.Msg {
	files, _ := filepath.Glob(filepath.Join(historyDir(), "*.json"))
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

// normalizeText flattens whitespace and caps the size of attachment text.
func normalizeText(content string) string {
	content = strings.Join(strings.Fields(content), " ")
	if len(content) > maxAttachChars {
		content = strings.ToValidUTF8(content[:maxAttachChars], "") + "\n...[TRUNCATED]..."
	}
	return content
}

func processFile(path string) tea.Msg {
	path = strings.TrimSpace(path)
	path = strings.Trim(path, `"'`)
	path = strings.ReplaceAll(path, `\ `, " ") // terminals escape spaces in dragged paths
	path = filepath.Clean(path)
	ext := strings.ToLower(filepath.Ext(path))

	var content string
	var err error
	isImage := false
	base64Str := ""

	switch {
	case textExts[ext]:
		content, err = readTextFile(path)
	case ext == ".pdf":
		content, err = readPDF(path)
	case ext == ".docx":
		var xmlStr string
		xmlStr, err = extractXMLFromZip(path, "word/document.xml")
		if err == nil {
			content = stripXML(xmlStr)
		}
	case ext == ".odt":
		var xmlStr string
		xmlStr, err = extractXMLFromZip(path, "content.xml")
		if err == nil {
			content = stripXML(xmlStr)
		}
	case ext == ".abw":
		var xmlStr string
		xmlStr, err = readTextFile(path)
		if err == nil {
			content = stripXML(xmlStr)
		}
	case ext == ".png", ext == ".jpg", ext == ".jpeg", ext == ".webp":
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
		content = normalizeText(content)
	} else {
		content = "[Image Data]"
	}

	return fileAttachedMsg{attachment: Attachment{Name: filepath.Base(path), Path: path, Text: content, IsImage: isImage, Base64: base64Str}}
}

// editInNano edits a *copy* of the file (FIXED: the original used to be
// modified in place, and nano on a PDF/DOCX just showed binary garbage).
func editInNano(path string, index int) tea.Cmd {
	data, err := os.ReadFile(path)
	if err != nil {
		return func() tea.Msg { return fileErrorMsg{err: err.Error()} }
	}
	tmp, err := os.CreateTemp("", "ollama-tui-*"+filepath.Ext(path))
	if err != nil {
		return func() tea.Msg { return fileErrorMsg{err: err.Error()} }
	}
	tmpPath := tmp.Name()
	_, werr := tmp.Write(data)
	tmp.Close()
	if werr != nil {
		os.Remove(tmpPath)
		return func() tea.Msg { return fileErrorMsg{err: werr.Error()} }
	}

	c := exec.Command("nano", tmpPath)
	return tea.ExecProcess(c, func(err error) tea.Msg {
		defer os.Remove(tmpPath)
		if err != nil {
			return fileErrorMsg{err: err.Error()}
		}
		content, err := readTextFile(tmpPath)
		if err != nil {
			return fileErrorMsg{err: err.Error()}
		}
		return fileEditedMsg{index: index, text: normalizeText(content)}
	})
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = msg.Width - 4
		m.viewport.Height = msg.Height - (m.headerLines() + 10)
		if m.viewport.Height < 3 {
			m.viewport.Height = 3
		}
		m.textarea.SetWidth(msg.Width - 8)
		m.promptInput.SetWidth(msg.Width - 14)
		m.fileInput.SetWidth(msg.Width - 14)
		m.hostInput.SetWidth(msg.Width - 14)
		w := msg.Width - 38
		if w < 10 {
			w = 10
		}
		for i := range m.settingInputs {
			m.settingInputs[i].SetWidth(w)
		}
		pw := msg.Width - 30
		if pw < 10 {
			pw = 10
		}
		for i := range m.pullInputs {
			m.pullInputs[i].SetWidth(pw)
		}
		kw := msg.Width - 30
		if kw < 10 {
			kw = 10
		}
		for i := range m.keyInputs {
			m.keyInputs[i].Width = kw
		}
		m.filterInput.Width = kw
		m.consoleVP.Width = msg.Width - 4
		m.consoleVP.Height = msg.Height - 8
		if m.consoleVP.Height < 3 {
			m.consoleVP.Height = 3
		}
		m.consoleIn.Width = kw
		m.refreshConsole()

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
		m.listError = ""
		// An empty library is fine now: the model screen lets you pull one.
		m.models = nil
		m.cursor = 0
		m.state = stateSelectModel
		m.statusMsg = "No models installed yet - press Ctrl+P to pull one."
		m.statusIsError = false

	case ollamaStartedMsg:
		if m.isCloud() {
			return m, nil // the user moved on to a cloud backend meanwhile
		}
		if msg.cmd != nil && m.ollamaCmd == nil {
			m.ollamaCmd = msg.cmd // so quitting can offer to stop it
		}
		if msg.missing {
			m.listError = "Ollama is not installed (or not in your PATH)."
			m.state = stateSelectModel
			return m, nil
		}
		cmds = append(cmds, fetchModels(m.host))
	case cloudModelsMsg:
		if msg.provider != m.backend {
			return m, nil // a late answer from a backend we already left
		}
		m.models = make([]string, 0, len(msg.models))
		m.cloudInfo = make(map[string]cloudModel, len(msg.models))
		m.listError = ""
		for _, cm := range msg.models {
			m.models = append(m.models, cm.ID)
			m.cloudInfo[cm.ID] = cm
		}
		if m.cursor >= len(m.models) {
			m.cursor = 0
		}
		m.state = stateSelectModel
		m.statusMsg = fmt.Sprintf("%s: %d models. Press / to filter.", backendLabel(m.backend), len(m.models))
		m.statusIsError = false
		cmds = append(cmds, clearStatusCmd(5*time.Second))

	case modelCapabilitiesMsg:
		m.applyCaps(msg)

	case fileAttachedMsg:
		if m.swapIndex == -1 {
			m.attachments = append(m.attachments, msg.attachment)
		} else {
			m.attachments[m.swapIndex] = msg.attachment
			m.swapIndex = -1
		}
		m.state = stateAttachmentList
		m.statusMsg = fmt.Sprintf("Attached: %s", msg.attachment.Name)
		cmds = append(cmds, clearStatusCmd(5*time.Second))

	case fileEditedMsg:
		if msg.index >= 0 && msg.index < len(m.attachments) {
			m.attachments[msg.index].Text = msg.text
			m.statusMsg = "File edited and reloaded."
			m.statusIsError = false
		}
		m.state = stateAttachmentList
		cmds = append(cmds, clearStatusCmd(5*time.Second))

	case fileErrorMsg:
		m.state = stateChat
		m.focusMode = "textarea"
		m.swapIndex = -1
		m.textarea.Focus()
		m.statusMsg = "File error: " + msg.err
		m.statusIsError = true
		cmds = append(cmds, clearStatusCmd(5*time.Second))

	case modelsFetchedMsg:
		if m.isCloud() {
			return m, nil // a late answer from the local server after switching away
		}
		m.models = msg.models
		m.listError = ""
		if m.selectAfterFetch != "" {
			for idx, name := range m.models {
				if name == m.selectAfterFetch {
					m.cursor = idx
				}
			}
			m.selectAfterFetch = ""
		}
		if m.cursor >= len(m.models) {
			m.cursor = 0
		}
		m.state = stateSelectModel

	case historyFilesMsg:
		m.historyFiles = msg.files
		m.historyCursor = 0
		m.state = stateHistory

	case streamChunkMsg:
		// FIXED: ignore chunks from a stream that was stopped or replaced, so they
		// can't leak into a different message/chat.
		if m.streamChan == nil || msg.ch != m.streamChan {
			return m, nil
		}
		if msg.Response != "" || msg.Thinking != "" || len(msg.ToolCalls) > 0 {
			m.gotOutput = true
		}
		if msg.Usage != nil {
			m.addUsage(*msg.Usage)
		}
		if len(msg.ToolCalls) > 0 {
			m.pendingCalls = append(m.pendingCalls, msg.ToolCalls...)
		}
		if last := len(m.messages) - 1; last >= 0 && m.messages[last].Role == "AI" {
			if msg.Thinking != "" {
				m.liveThinkingBuilder.WriteString(msg.Thinking)
				m.messages[last].Thinking = m.liveThinkingBuilder.String()
			}
			if msg.Response != "" {
				m.liveBuilder.WriteString(msg.Response)
				m.messages[last].Content = m.liveBuilder.String()
			}
			if msg.Error != "" {
				errText := "⚠️ Error: " + msg.Error
				if m.messages[last].Content == "" && m.messages[last].Thinking == "" {
					// Nothing was generated: turn the empty AI bubble into a note.
					m.messages[last].Role = "System"
					m.messages[last].Content = errText
				} else {
					m.messages = append(m.messages, ChatMessage{Role: "System", Content: errText})
				}
				m.statusMsg = msg.Error
				m.statusIsError = true
				cmds = append(cmds, clearStatusCmd(5*time.Second))
			}
		}
		// Redrawing (markdown!) on every single chunk gets slower and slower as the
		// reply grows, so the text is drawn by the next render tick instead.
		m.dirty = true
		cmds = append(cmds, waitForStream(m.streamChan))

	case pullCheckedMsg:
		// Ignore answers to a check the user has already left.
		if m.state != stateModelPull || m.pullStep != pullChecking || msg.ref != m.pullModel {
			return m, nil
		}
		switch {
		case msg.err != nil:
			m.pullStep = pullInput
			m.pullErr = msg.err.Error()
			m.focusPull()
		case msg.notFound:
			m.pullStep = pullInput
			m.pullErr = fmt.Sprintf("%s was not found in the Ollama library.", msg.ref)
			m.focusPull()
		default:
			m.pullStep = pullConfirm
			m.pullSize = msg.size
		}

	case pullProgressMsg:
		if m.pullChan == nil || msg.ch != m.pullChan {
			return m, nil
		}
		ev := msg.ev
		if ev.Err != "" {
			if m.pullCancel != nil {
				m.pullCancel()
			}
			m.pullChan = nil
			m.pullStep = pullInput
			m.pullErr = ev.Err
			m.focusPull()
			return m, nil
		}
		if ev.Status != "" {
			m.pullStatus = ev.Status
		}
		if ev.Status == "success" {
			m.pullSuccess = true
		}
		if ev.Total > 0 {
			m.pullTotal = ev.Total
			if ev.Completed > m.pullDone {
				m.pullDone = ev.Completed
			}
		}
		cmds = append(cmds, waitForPull(m.pullChan))

	case pullDoneMsg:
		if m.pullChan == nil || msg.ch != m.pullChan {
			return m, nil
		}
		m.pullChan = nil
		if m.pullCancel != nil {
			m.pullCancel()
		}
		if m.pullSuccess {
			m.selectAfterFetch = m.pullModel
			m.state = stateSelectModel
			m.statusMsg = "Downloaded " + m.pullModel
			m.statusIsError = false
			cmds = append(cmds, fetchModels(m.host), clearStatusCmd(5*time.Second))
		} else {
			m.pullStep = pullInput
			m.pullErr = "The download stopped before it finished. Try again, it resumes where it left off."
			m.focusPull()
		}

	case deleteDoneMsg:
		if msg.err != nil {
			m.statusMsg = "Delete failed: " + msg.err.Error()
			m.statusIsError = true
		} else {
			m.statusMsg = "Deleted " + msg.name
			m.statusIsError = false
			cmds = append(cmds, fetchModels(m.host))
		}
		cmds = append(cmds, clearStatusCmd(5*time.Second))
	case renderTickMsg:
		// Draw the streamed text at most ten times a second.
		if m.streamChan == nil || msg.ch != m.streamChan {
			return m, nil // the final redraw happens when the stream ends
		}
		if m.dirty {
			m.dirty = false
			m.updateViewport()
		}
		cmds = append(cmds, renderTickCmd(m.streamChan))

	case consoleOutMsg:
		m.consolePrint(msg.lines...)

	case consoleKilledMsg:
		m.ollamaCmd = nil
		m.consolePrint(msg.lines...)
	case consoleChatChunkMsg:
		if m.ccChan == nil || msg.ch != m.ccChan {
			return m, nil
		}
		if msg.Error != "" {
			m.consolePrint("✗ " + msg.Error)
			m.endConsoleChat()
			return m, nil
		}
		if m.ccFirst.IsZero() && (msg.Response != "" || msg.Thinking != "") {
			m.ccFirst = time.Now()
		}
		m.ccText += msg.Response
		m.ccThinking += msg.Thinking
		if msg.Usage != nil {
			m.ccUsage = msg.Usage
		}
		m.refreshConsole()
		cmds = append(cmds, waitForConsoleChat(m.ccChan))

	case consoleChatDoneMsg:
		if m.ccChan != nil && msg.ch == m.ccChan {
			m.finishConsoleChat()
		}

	case consoleStartedMsg:
		if msg.cmd != nil && m.ollamaCmd == nil {
			m.ollamaCmd = msg.cmd
		}
		m.consolePrint(msg.lines...)
		// Only refresh the model list when the model screen is what is showing.
		if msg.ready && !m.isCloud() && (m.state == stateSelectModel || m.state == stateLoading) {
			cmds = append(cmds, fetchModels(m.host))
		}

	case statsTickMsg:
		if m.showStats && msg.gen == m.statsGen {
			cmds = append(cmds, collectStatsCmd(m.cpuPrev, m.statsGen))
		}

	case statsMsg:
		if !m.showStats || msg.gen != m.statsGen {
			return m, nil
		}
		m.stats = msg.stats
		m.cpuPrev = msg.cpu
		m.haveStats = true
		cmds = append(cmds, statsTickCmd(m.statsGen))
	case tickMsg:
		// Keep ticking only for the stream that is still running. Returning from
		// Update is enough to make Bubble Tea redraw the status bar.
		if m.isThinking && m.streamChan != nil && msg.ch == m.streamChan {
			cmds = append(cmds, tickCmd(m.streamChan))
		}

	case streamDoneMsg:
		if m.streamChan != nil && msg.ch == m.streamChan {
			m.isThinking = false
			m.streamChan = nil
			if m.cancelStream != nil {
				m.cancelStream()
			}
			m.dirty = false
			if len(m.pendingCalls) > 0 {
				// The model asked for tools: run them and let it carry on.
				toolCmd := m.runPendingTools()
				cmds = append(cmds, toolCmd)
			} else if last := len(m.messages) - 1; last >= 0 && m.messages[last].Role == "AI" {
				// Print how long the reply took under it.
				m.messages[last].Elapsed = formatDuration(time.Since(m.genStart))
			}
			m.updateViewport()
			if m.autoScroll {
				m.viewport.GotoBottom()
			}
		}

	case errMsg:
		// Instead of crashing, show the error and return to the previous state
		m.statusMsg = msg.err.Error()
		m.statusIsError = true
		m.models = nil // the old list may belong to a server we can no longer reach
		m.cloudInfo = nil
		m.listError = msg.err.Error()
		m.state = stateSelectModel // Go back to model selection so they can fix the host
		cmds = append(cmds, clearStatusCmd(5*time.Second))
		return m, tea.Batch(cmds...)

	case tea.KeyMsg:
		if m.state == stateError {
			if msg.String() == "ctrl+c" || msg.String() == "enter" || msg.String() == "esc" {
				if !m.keepServer {
					stopOllama(m.ollamaCmd)
				}
				return m, tea.Quit
			}
			return m, nil
		}

		if msg.String() == "ctrl+c" {
			m.console = false // so the exit question below is visible
			if m.pullCancel != nil {
				m.pullCancel()
			}
			// FIXED: only ask about stopping the server if we actually started one.
			if m.ollamaCmd != nil && (m.state == stateChat || m.state == stateHistory || m.state == stateSettings || m.state == stateFileAttach || m.state == stateAttachmentList || m.state == stateHostInput) {
				m.state = stateExiting
				m.textarea.Blur()
				m.promptInput.Blur()
				m.fileInput.Blur()
				m.hostInput.Blur()
			} else {
				return m, tea.Quit
			}
		}

		// Ctrl+G opens the console from anywhere.
		if msg.String() == "ctrl+g" && !m.console && m.state != stateExiting {
			m.openConsole()
			return m, nil
		}
		// Alt+H shows or hides the CPU / RAM / GPU percentages in the header.
		if msg.String() == "alt+h" && m.state == stateChat && !m.console {
			statsCmd := m.toggleStats()
			return m, statsCmd
		}
		if m.console {
			consoleCmd := m.updateConsoleKey(msg)
			return m, consoleCmd
		}
		if m.state == stateExiting {
			if msg.String() == "y" || msg.String() == "Y" {
				stopOllama(m.ollamaCmd)
				return m, tea.Quit
			}
			if msg.String() == "n" || msg.String() == "N" {
				return m, tea.Quit
			}
			return m, nil
		}

		if m.state == stateHostInput {
			switch msg.String() {
			case "esc":
				m.state = stateSelectModel
				m.hostInput.Blur()
				return m, nil
			case "enter":
				if newHost := normalizeHost(m.hostInput.Value()); newHost != "" {
					m.host = newHost
				}
				m.state = stateLoading
				m.hostInput.Blur()
				return m, fetchModels(m.host)
			}
			var cmd tea.Cmd
			m.hostInput, cmd = m.hostInput.Update(msg)
			return m, cmd
		}

		if m.state == stateFileAttach {
			switch msg.String() {
			case "esc":
				m.state = stateChat
				m.focusMode = "textarea"
				m.swapIndex = -1
				m.fileInput.Reset()
				m.fileInput.Blur()
				m.textarea.Focus()
				m.statusMsg = "Attachment cancelled."
				m.statusIsError = false
				return m, clearStatusCmd(5 * time.Second)
			case "enter":
				path := strings.TrimSpace(m.fileInput.Value())
				m.fileInput.Reset()
				m.fileInput.Blur()
				if path != "" {
					return m, func() tea.Msg { return processFile(path) }
				}
				m.state = stateChat
				m.focusMode = "textarea"
				m.swapIndex = -1
				m.textarea.Focus()
				return m, nil
			}
			var cmd tea.Cmd
			m.fileInput, cmd = m.fileInput.Update(msg)
			return m, cmd
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
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				}
			case "S":
				if len(m.attachments) > 0 {
					m.swapIndex = m.attachCursor
					m.state = stateFileAttach
					m.fileInput.Focus()
				}
			case "N":
				if len(m.attachments) > 0 {
					a := m.attachments[m.attachCursor]
					if a.IsImage || !textExts[strings.ToLower(filepath.Ext(a.Path))] {
						m.statusMsg = "Editing is only available for plain-text attachments."
						m.statusIsError = true
						cmds = append(cmds, clearStatusCmd(5*time.Second))
					} else {
						cmds = append(cmds, editInNano(a.Path, m.attachCursor))
					}
				}
			}
			return m, tea.Batch(cmds...)
		}

		if m.state == stateSettings {
			next, swallow, open := konamiStep(m.konami, msg.String())
			m.konami = next
			if open {
				m.openConsole()
				return m, nil
			}
			if swallow {
				return m, nil
			}
			switch msg.String() {
			case "esc":
				m.closeSettings()
				m.statusMsg = "Settings cancelled."
				m.statusIsError = false
				return m, clearStatusCmd(5 * time.Second)
			case "ctrl+s":
				cmd := m.saveSettings()
				return m, cmd
			case "tab", "shift+tab":
				if msg.String() == "tab" {
					m.settingFocus = (m.settingFocus + 1) % 5
				} else {
					m.settingFocus = (m.settingFocus + 4) % 5
				}
				m.focusSetting()
				return m, nil
			case "enter":
				// In the system prompt Enter is a newline. On the other fields it moves
				// to the next one, and on the last field it saves.
				if m.settingFocus == 4 {
					cmd := m.saveSettings()
					return m, cmd
				}
				if m.settingFocus != 0 {
					m.settingFocus++
					m.focusSetting()
					return m, nil
				}
			}
			var cmd tea.Cmd
			if m.settingFocus == 0 {
				m.promptInput, cmd = m.promptInput.Update(msg)
			} else {
				i := m.settingFocus - 1
				m.settingInputs[i], cmd = m.settingInputs[i].Update(msg)
			}
			return m, cmd
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
					// FIXED: decode into a fresh slice. Decoding into m.messages reused the
					// old backing array, so omitted fields (thinking/context) leaked from
					// the previous chat into the loaded one.
					var loaded []ChatMessage
					if err == nil {
						err = json.Unmarshal(data, &loaded)
					}
					m.state = stateChat
					m.focusMode = "list"
					if err != nil {
						m.statusMsg = "Failed to load history: " + err.Error()
						m.statusIsError = true
					} else {
						if len(loaded) == 0 {
							loaded = []ChatMessage{{Role: "System", Content: "Loaded an empty history."}}
						}
						m.messages = loaded
						m.renderedCache = nil
						m.selectedMsg = len(m.messages) - 1
						m.autoScroll = true
						m.updateViewport()
						m.viewport.GotoBottom()
						m.statusMsg = "Loaded history."
						m.statusIsError = false
					}
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				}
			}
			return m, tea.Batch(cmds...)
		}

		if m.state == stateAPIKeys {
			switch msg.String() {
			case "esc":
				m.closeKeys()
				return m, nil
			case "ctrl+s":
				cmd := m.saveKeys()
				return m, cmd
			case "tab", "shift+tab", "up", "down":
				m.keyFocus = 1 - m.keyFocus
				m.focusKey()
				return m, nil
			case "enter":
				if m.keyFocus == 1 {
					cmd := m.saveKeys()
					return m, cmd
				}
				m.keyFocus = 1
				m.focusKey()
				return m, nil
			}
			var cmd tea.Cmd
			m.keyInputs[m.keyFocus], cmd = m.keyInputs[m.keyFocus].Update(msg)
			return m, cmd
		}
		if m.state == stateModelPull {
			switch m.pullStep {
			case pullChecking:
				if msg.String() == "esc" {
					m.pullStep = pullInput
					m.focusPull()
				}
				return m, nil
			case pullConfirm:
				switch msg.String() {
				case "y", "Y":
					ctx, cancel := context.WithCancel(context.Background())
					ch := make(chan pullEvent)
					m.pullCancel = cancel
					m.pullChan = ch
					m.pullStep = pullDownloading
					m.pullDone = 0
					m.pullTotal = m.pullSize
					m.pullStatus = "starting"
					m.pullSuccess = false
					m.pullStart = time.Now()
					return m, tea.Batch(startPull(ctx, m.host, m.pullModel, m.pullSize, ch), waitForPull(ch))
				case "n", "N", "esc":
					m.pullStep = pullInput
					m.focusPull()
				}
				return m, nil
			case pullDownloading:
				if msg.String() == "esc" {
					if m.pullCancel != nil {
						m.pullCancel()
					}
					m.pullChan = nil // ignore anything still in flight
					m.pullStep = pullInput
					m.pullErr = "Download cancelled. What was already downloaded is kept and resumes next time."
					m.focusPull()
				}
				return m, nil
			}

			// pullInput: typing the model name and the parameters/tag
			switch msg.String() {
			case "esc":
				for k := range m.pullInputs {
					m.pullInputs[k].Blur()
				}
				m.state = stateSelectModel
				return m, nil
			case "tab", "shift+tab", "up", "down":
				m.pullFocus = 1 - m.pullFocus
				m.focusPull()
				return m, nil
			case "enter":
				if m.pullFocus == 0 {
					m.pullFocus = 1
					m.focusPull()
					return m, nil
				}
				name := strings.ToLower(strings.TrimSpace(m.pullInputs[0].Value()))
				tag := strings.ToLower(strings.TrimSpace(m.pullInputs[1].Value()))
				if tag == "" {
					tag = "latest"
				}
				if !validModelPart(name, true) || !validModelPart(tag, false) {
					m.pullErr = "Names can only use letters, numbers and . _ - (plus / for user models)."
					return m, nil
				}
				m.pullErr = ""
				m.pullModel = name + ":" + tag
				m.pullStep = pullChecking
				return m, checkRegistry(m.pullModel)
			}
			var cmd tea.Cmd
			m.pullInputs[m.pullFocus], cmd = m.pullInputs[m.pullFocus].Update(msg)
			return m, cmd
		}

		if m.state == stateSelectModel {
			// A delete is waiting for y/n.
			if m.deleteTarget != "" {
				switch msg.String() {
				case "y", "Y":
					name := m.deleteTarget
					m.deleteTarget = ""
					return m, deleteModel(m.host, name)
				case "n", "N", "esc":
					m.deleteTarget = ""
				}
				return m, nil
			}
			// Typing in the filter box.
			if m.filterMode {
				switch msg.String() {
				case "esc":
					m.filterMode = false
					m.filterInput.Blur()
					m.filterInput.SetValue("")
					m.filter = ""
					m.moveCursor(0)
					return m, nil
				case "enter":
					m.filterMode = false
					m.filterInput.Blur()
					return m, nil
				case "up":
					m.moveCursor(-1)
					return m, nil
				case "down":
					m.moveCursor(1)
					return m, nil
				}
				var cmd tea.Cmd
				m.filterInput, cmd = m.filterInput.Update(msg)
				m.filter = m.filterInput.Value()
				m.moveCursor(0)
				return m, cmd
			}
			switch msg.String() {
			case "up", "k":
				m.moveCursor(-1)
			case "down", "j":
				m.moveCursor(1)
			case ":":
				m.openConsole()
				return m, nil
			case "/":
				m.filterMode = true
				m.filterInput.SetValue(m.filter)
				m.filterInput.Focus()
				return m, nil
			case "esc":
				if m.filter != "" {
					m.filter = ""
					m.filterInput.SetValue("")
					m.moveCursor(0)
				}
			case "ctrl+b":
				// Ollama -> OpenRouter -> OpenAI -> Ollama ...
				cmd := m.switchBackend()
				return m, cmd
			case "ctrl+r":
				// Try loading the list again: server started, key added, network back...
				cmd := m.reloadModels()
				return m, cmd
			case "ctrl+k":
				m.openKeys()
				return m, nil
			case "ctrl+p":
				if m.isCloud() {
					cmd := m.localOnlyStatus()
					return m, cmd
				}
				m.openPull()
				return m, nil
			case "delete", "d":
				// Ask first.
				if m.isCloud() {
					cmd := m.localOnlyStatus()
					return m, cmd
				}
				m.moveCursor(0)
				if len(m.visibleModels()) > 0 {
					m.deleteTarget = m.models[m.cursor]
				}
			case "shift+delete", "D":
				// No questions asked.
				if m.isCloud() {
					cmd := m.localOnlyStatus()
					return m, cmd
				}
				m.moveCursor(0)
				if len(m.visibleModels()) > 0 {
					return m, deleteModel(m.host, m.models[m.cursor])
				}
			case "ctrl+h":
				if m.isCloud() {
					cmd := m.localOnlyStatus()
					return m, cmd
				}
				m.state = stateHostInput
				m.hostInput.SetValue(m.host)
				m.hostInput.Focus()
				return m, nil
			case "enter":
				m.moveCursor(0) // make sure the selection is one the filter shows
				if len(m.models) == 0 || len(m.visibleModels()) == 0 {
					return m, nil // nothing to select (e.g. the host was unreachable)
				}
				// Cloud models need a key before the first message.
				if m.isCloud() && m.apiKey() == "" {
					m.openKeys()
					m.statusMsg = "Add your " + backendLabel(m.backend) + " API key first."
					m.statusIsError = true
					return m, clearStatusCmd(5 * time.Second)
				}
				m.state = stateChat
				m.focusMode = "textarea"
				m.selectedMsg = -1
				m.autoScroll = true
				m.filter = ""
				m.filterMode = false
				m.filterInput.SetValue("")

				if m.isSwitchingModel {
					m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("Switched model to: %s", m.models[m.cursor])})
					m.isSwitchingModel = false
				} else if len(m.messages) == 0 {
					m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("Loaded model: %s", m.models[m.cursor])})
				}

				m.updateViewport()
				m.textarea.Focus()
				if m.isCloud() {
					m.applyCaps(m.cloudCaps(m.models[m.cursor]))
				} else {
					cmds = append(cmds, fetchCapabilities(m.host, m.models[m.cursor]))
				}
			}
			return m, tea.Batch(cmds...)
		}

		if m.state == stateChat {
			if msg.String() == "alt+s" && m.isThinking {
				if m.cancelStream != nil {
					m.cancelStream()
				}
				m.streamChan = nil // anything still in flight is ignored
				m.pendingCalls = nil
				m.isThinking = false
				if last := len(m.messages) - 1; last >= 0 && m.messages[last].Role == "AI" {
					m.messages[last].Content += " [Stopped by user]"
					m.messages[last].Elapsed = formatDuration(time.Since(m.genStart))
				}
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
					cmds = append(cmds, clearStatusCmd(3*time.Second))
				} else {
					m.statusMsg = "Sorry, this model does not support Thinking."
					m.statusIsError = true
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				}
				return m, tea.Batch(cmds...)
			}
			if msg.String() == "ctrl+y" {
				if m.supportsTools {
					m.useTools = !m.useTools
					m.statusMsg = fmt.Sprintf("Tools: %v", m.useTools)
					cmds = append(cmds, clearStatusCmd(3*time.Second))
				} else {
					m.statusMsg = "Sorry, this model does not support Tools."
					m.statusIsError = true
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				}
				return m, tea.Batch(cmds...)
			}

			if msg.String() == "alt+t" {
				m.showThinking = !m.showThinking
				m.statusMsg = fmt.Sprintf("Show Thinking: %v", m.showThinking)
				cmds = append(cmds, clearStatusCmd(3*time.Second))
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
					m.autoScroll = m.viewport.AtBottom() // back at the bottom: follow the reply again
				case "C":
					if m.selectedMsg >= 0 && m.selectedMsg < len(m.messages) {
						if err := clipboard.WriteAll(m.messages[m.selectedMsg].Content); err != nil {
							m.statusMsg = "Copy failed: " + err.Error() + " (is xclip installed?)"
							m.statusIsError = true
						} else {
							m.statusMsg = "Copied to clipboard!"
							m.statusIsError = false
						}
						cmds = append(cmds, clearStatusCmd(5*time.Second))
					}
				// FIXED: N / S / B / Ctrl+O change the conversation, so they are blocked while
				// a reply is streaming (late chunks used to land in the wrong chat).
				case "N":
					if m.isThinking {
						cmds = append(cmds, m.setBusyStatus())
						break
					}
					m.messages = []ChatMessage{{Role: "System", Content: fmt.Sprintf("Started new chat with %s", m.models[m.cursor])}}
					m.selectedMsg = 0
					m.autoScroll = true
					m.updateViewport()
					m.statusMsg = "Started new chat."
					m.statusIsError = false
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				case "S":
					if m.isThinking {
						cmds = append(cmds, m.setBusyStatus())
						break
					}
					m.state = stateSelectModel
					m.isSwitchingModel = true
				case ":":
					m.openConsole()
				case "P":
					m.openSettings()
				case "J":
					dir := historyDir()
					filename := fmt.Sprintf("chat_%s.json", time.Now().Format("20060102_150405"))
					data, _ := json.MarshalIndent(m.messages, "", "  ")
					err := os.MkdirAll(dir, 0755)
					if err == nil {
						err = os.WriteFile(filepath.Join(dir, filename), data, 0644)
					}
					if err != nil {
						m.statusMsg = "Save failed: " + err.Error()
						m.statusIsError = true
					} else {
						m.statusMsg = "Saved: " + filename
						m.statusIsError = false
					}
					cmds = append(cmds, clearStatusCmd(5*time.Second))
				case "B":
					if m.isThinking {
						cmds = append(cmds, m.setBusyStatus())
						break
					}
					if m.selectedMsg >= 0 && m.selectedMsg < len(m.messages) {
						m.messages = m.messages[:m.selectedMsg+1]
						m.messages = append(m.messages, ChatMessage{Role: "System", Content: "--- Branched from here ---"})
						m.selectedMsg = len(m.messages) - 1
						m.autoScroll = true
						m.updateViewport()
						m.statusMsg = "Branched chat."
						m.statusIsError = false
						cmds = append(cmds, clearStatusCmd(5*time.Second))
					}
				case "ctrl+o":
					if m.isThinking {
						cmds = append(cmds, m.setBusyStatus())
						break
					}
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

				// FIXED: handle Enter *before* the textarea sees it. Before, the textarea
				// inserted a newline first, so every prompt was sent with a trailing "\n".
				// Alt+Enter and Ctrl+J (and Shift+Enter, in terminals that report it) add a new line.
				if isNewlineKey(msg.String()) {
					m.textarea, cmd = m.textarea.Update(tea.KeyMsg{Type: tea.KeyEnter})
					cmds = append(cmds, cmd)
					return m, tea.Batch(cmds...)
				}
				if msg.String() == "enter" {
					userPrompt := strings.TrimSpace(m.textarea.Value())
					if userPrompt == "" {
						return m, nil
					}

					// Text files become context for this message; images go along as images.
					var attachCtx strings.Builder
					var imageB64 []string
					for _, a := range m.attachments {
						if a.IsImage {
							imageB64 = append(imageB64, a.Base64)
						} else {
							attachCtx.WriteString(fmt.Sprintf("--- %s ---\n%s\n\n", a.Name, a.Text))
						}
					}
					ctxText := ""
					if attachCtx.Len() > 0 {
						ctxText = "Context from attached files:\n" + attachCtx.String()
					}

					m.messages = append(m.messages, ChatMessage{Role: "You", Content: userPrompt, Context: ctxText})
					for _, a := range m.attachments {
						if a.IsImage {
							m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("🖼️ Image: %s", a.Name)})
						} else {
							m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("📎 Attached: %s", a.Name)})
						}
					}

					// FIXED: the whole conversation is sent, not just the last prompt.
					// Built here (on the UI goroutine) so the streaming goroutine never
					// touches m.messages while it is being updated.
					m.textarea.Reset()
					m.attachments = nil

					// Everything the model should see is built inside beginReply.
					replyCmd := m.beginReply(imageB64, true, m.useTools)
					cmds = append(cmds, replyCmd)
					return m, tea.Batch(cmds...)
				}

				m.textarea, cmd = m.textarea.Update(msg)
				cmds = append(cmds, cmd)
			}
		}
	}

	return m, tea.Batch(cmds...)
}

func clearStatusCmd(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(time.Time) tea.Msg { return clearStatusMsg{} })
}

func (m *model) setBusyStatus() tea.Cmd {
	m.statusMsg = "Wait for the AI to finish (or press Alt+S to stop it)."
	m.statusIsError = true
	return clearStatusCmd(4 * time.Second)
}

// statusLine renders the current status message (used by the non-chat screens).
func (m model) statusLine() string {
	if m.statusMsg == "" {
		return ""
	}
	color := colorGreen
	if m.statusIsError {
		color = colorRed
	}
	return lipgloss.NewStyle().Foreground(color).Render(m.statusMsg)
}

// ollamaStartedMsg is the result of startOllama.
type ollamaStartedMsg struct {
	cmd     *exec.Cmd // the server we started (nil if one was already running)
	missing bool      // ollama is not installed
}

// startOllama makes sure a local Ollama server is answering, starting one if
// it is not (only for the default local address).
func startOllama(host string) tea.Cmd {
	return func() tea.Msg {
		if host != defaultHost || ollamaReachable(host) {
			return ollamaStartedMsg{}
		}
		path, err := exec.LookPath("ollama")
		if err != nil {
			return ollamaStartedMsg{missing: true}
		}
		cmd := exec.Command(path, "serve")
		if err := cmd.Start(); err != nil {
			return ollamaStartedMsg{}
		}
		for i := 0; i < 20 && !ollamaReachable(host); i++ {
			time.Sleep(250 * time.Millisecond)
		}
		return ollamaStartedMsg{cmd: cmd}
	}
}

// pageStyle frames the full-screen views. FIXED: it now has a width, so long
// help lines wrap at the window edge instead of running off-screen.
func (m model) pageStyle() lipgloss.Style {
	s := lipgloss.NewStyle().Margin(1, 2)
	if m.width > 8 {
		s = s.Width(m.width - 4)
	}
	return s
}

// Optimized updateViewport with caching
func (m *model) updateViewport() {
	if m.cacheWidth != m.viewport.Width || m.cacheShowThinking != m.showThinking || len(m.renderedCache) != len(m.messages) {
		// Everything has to be drawn again.
		m.renderedCache = make([]string, len(m.messages))
	} else if m.cacheSelected != m.selectedMsg || m.cacheFocusMode != m.focusMode {
		// Only the old and the new selection look different.
		for _, k := range []int{m.cacheSelected, m.selectedMsg} {
			if k >= 0 && k < len(m.renderedCache) {
				m.renderedCache[k] = ""
			}
		}
	}
	m.cacheWidth = m.viewport.Width
	m.cacheSelected = m.selectedMsg
	m.cacheFocusMode = m.focusMode
	m.cacheShowThinking = m.showThinking

	textWidth := m.viewport.Width - 3
	var sb strings.Builder
	selectedMsgStartLine := 0
	selectedMsgEndLine := 0
	currentLine := 0

	for i, msg := range m.messages {
		if len(m.renderedCache[i]) == 0 || i == len(m.messages)-1 {
			m.renderedCache[i] = m.renderMessage(msg, i, textWidth)
		}

		lines := strings.Count(m.renderedCache[i], "\n") + 1
		if i == m.selectedMsg {
			selectedMsgStartLine = currentLine
			selectedMsgEndLine = currentLine + lines - 1
		}
		currentLine += lines

		sb.WriteString(m.renderedCache[i])
		sb.WriteString("\n")

		// A blank line after every chat message (small notes stay packed together).
		if !isNoteMsg(msg) && i < len(m.messages)-1 {
			sb.WriteString("\n")
			currentLine++
		}
	}

	m.viewport.SetContent(sb.String())

	switch {
	case m.focusMode == "list" && m.selectedMsg >= 0 && m.selectedMsg != m.snappedMsg:
		// The selection moved: bring that message into view. (Doing this on every
		// update undid the user's scrolling while a reply was streaming in.)
		m.snappedMsg = m.selectedMsg
		if m.autoScroll && m.selectedMsg == len(m.messages)-1 {
			m.viewport.GotoBottom() // following the newest message: go all the way down
		} else if selectedMsgStartLine < m.viewport.YOffset {
			m.viewport.SetYOffset(selectedMsgStartLine)
		} else if selectedMsgEndLine >= m.viewport.YOffset+m.viewport.Height {
			m.viewport.SetYOffset(selectedMsgEndLine - m.viewport.Height + 1)
		}
	case m.autoScroll:
		m.viewport.GotoBottom()
	}
	if m.selectedMsg < 0 || m.focusMode != "list" {
		m.snappedMsg = -1
	}
}

// isNoteMsg tells the small grey one-line entries (loaded model, attachments,
// errors, tool calls and results) from real chat messages.
func isNoteMsg(msg ChatMessage) bool {
	return msg.Role == "System" || msg.Role == "Tool" ||
		(msg.Role == "AI" && strings.TrimSpace(msg.Content) == "" && len(msg.ToolCalls) > 0)
}

func (m *model) renderMessage(msg ChatMessage, i int, textWidth int) string {
	isSelected := (i == m.selectedMsg && m.focusMode == "list")

	pointer := "   "
	if isSelected {
		pointer = lipgloss.NewStyle().Foreground(colorCyan).Bold(true).Render(" > ")
	}

	// Notes: a label and one wrapped line of text.
	noteLabel, noteText := "", ""
	switch {
	case msg.Role == "System":
		noteLabel, noteText = "•", msg.Content
	case msg.Role == "Tool":
		noteLabel, noteText = "🔧", shorten(msg.ToolName+" → "+msg.Content, 300)
	case isNoteMsg(msg):
		noteLabel, noteText = "🔧", "calling "+describeCalls(msg.ToolCalls)
	}
	isNote := noteLabel != ""

	// One colour per role: cyan for you, purple for the AI, grey for notes.
	roleColor := colorCyan
	label := "You"
	switch {
	case isNote:
		roleColor = colorGray
		label = noteLabel
	case msg.Role == "AI":
		roleColor = colorPurple
		label = "AI"
	}

	var styledLabel string
	switch {
	case isSelected:
		styledLabel = lipgloss.NewStyle().Background(roleColor).Foreground(lipgloss.Color("0")).Bold(true).Render(" " + label + " ")
	case isNote:
		styledLabel = lipgloss.NewStyle().Foreground(colorGray).Italic(true).Render(label)
	default:
		styledLabel = lipgloss.NewStyle().Foreground(roleColor).Bold(true).Render(label)
	}

	if isNote {
		labelWidth := lipgloss.Width(styledLabel)
		w := textWidth - labelWidth - 1
		if w < 10 {
			w = 10
		}
		lines := strings.Split(lipgloss.NewStyle().Width(w).Render(noteText), "\n")
		out := []string{styledLabel + " " + lines[0]}
		indent := strings.Repeat(" ", labelWidth+1)
		for _, l := range lines[1:] {
			out = append(out, indent+l)
		}
		return lipgloss.JoinHorizontal(lipgloss.Top, pointer, strings.Join(out, "\n"))
	}

	// You / AI: the coloured label gets its own line, the text sits next to a coloured bar.
	wrapWidth := textWidth - 2
	if wrapWidth < 10 {
		wrapWidth = 10
	}

	var body strings.Builder

	if msg.Role == "AI" && msg.Thinking != "" && m.showThinking {
		thoughtStyle := lipgloss.NewStyle().Foreground(colorGray).Italic(true)
		wrappedThoughts := lipgloss.NewStyle().Width(wrapWidth).Render(msg.Thinking)
		body.WriteString(thoughtStyle.Render("🤔 [Thinking...]"))
		body.WriteString("\n")
		body.WriteString(thoughtStyle.Render(wrappedThoughts))
		body.WriteString("\n\n")
	}

	var renderedText string
	if (msg.Role == "AI" || msg.Role == "You") && m.renderer != nil {
		out, err := m.renderer.Render(msg.Content)
		if err == nil {
			renderedText = strings.Trim(out, "\n")
		} else {
			renderedText = lipgloss.NewStyle().Width(wrapWidth).Render(msg.Content)
		}
	} else {
		renderedText = lipgloss.NewStyle().Width(wrapWidth).Render(msg.Content)
	}
	body.WriteString(renderedText)
	if msg.Role == "AI" && msg.Elapsed != "" {
		body.WriteString("\n")
		body.WriteString(lipgloss.NewStyle().Foreground(colorGray).Italic(true).Render("⏱️ " + msg.Elapsed))
	}

	bar := lipgloss.NewStyle().Foreground(roleColor).Render("▎")
	lines := append([]string{styledLabel}, strings.Split(body.String(), "\n")...)
	for k := range lines {
		lines[k] = bar + " " + lines[k]
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, pointer, strings.Join(lines, "\n"))
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
	if m.console {
		return m.consoleView()
	}
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
		return m.pageStyle().Render(content)
	}

	if m.state == stateFileAttach {
		title := "Attach File (Ctrl+F)"
		if m.swapIndex != -1 {
			title = "Replace Attachment (Shift+S)"
		}
		box := textareaStyle.Width(m.width - 10).Render(m.fileInput.View())
		help := lipgloss.NewStyle().Foreground(colorGray).Render("Drag & drop a file or type a path. Press Enter to attach, Esc to cancel.")
		content := lipgloss.JoinVertical(lipgloss.Left, aiLabelStyle.Render(title), "", box, "", help)
		return m.pageStyle().Render(content)
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
		if sl := m.statusLine(); sl != "" {
			title += "\n\n" + sl
		}
		return m.pageStyle().Render(title)
	}

	if m.state == stateSettings {
		label := func(i int, name string) string {
			if m.settingFocus == i {
				return aiLabelStyle.Render(name)
			}
			return lipgloss.NewStyle().Foreground(colorGray).Render(name)
		}

		rows := []string{
			aiLabelStyle.Render("Settings"),
			"",
			label(0, "System prompt"),
			textareaStyle.Width(m.width - 10).Render(m.promptInput.View()),
			"",
		}
		names := [4]string{"Temperature (0-2)", "Top P (0-1)", "Top K (1+)", "Context tokens (512+)"}
		for i, name := range names {
			rows = append(rows, lipgloss.JoinHorizontal(lipgloss.Top, label(i+1, fmt.Sprintf("%-24s", name)), m.settingInputs[i].View()))
		}

		note := "Leave a field blank to use the model's default."
		if m.modelMaxCtx > 0 {
			note += fmt.Sprintf(" This model supports up to %d context tokens.", m.modelMaxCtx)
		}
		help := lipgloss.NewStyle().Foreground(colorGray).Render(note + "\nTab / Shift+Tab: switch field • Enter: next field (saves on the last) • Ctrl+S: save • Ctrl+G: console • Esc: cancel")
		rows = append(rows, "", help)
		if sl := m.statusLine(); sl != "" {
			rows = append(rows, "", sl)
		}
		return m.pageStyle().Render(lipgloss.JoinVertical(lipgloss.Left, rows...))
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
		return m.pageStyle().Render(title)
	}

	if m.state == stateAPIKeys {
		return m.keysScreen()
	}
	if m.state == stateModelPull {
		return m.pullScreen()
	}
	if m.state == stateSelectModel {
		title := "Select a model • " + backendLabel(m.backend)
		if m.isCloud() {
			title += fmt.Sprintf(" • %d models", len(m.models))
		}
		title += "\n"
		if m.isCloud() {
			title += "Ctrl+B backend • Ctrl+K API keys • Ctrl+R reload • / filter • : console\n"
		} else {
			title += "Ctrl+B backend • Ctrl+K API keys • Ctrl+R reload • / filter • : console • Ctrl+H host • Ctrl+P pull • Del delete\n"
		}
		if m.filterMode {
			title += "Filter: " + m.filterInput.View() + "\n"
		} else if m.filter != "" {
			title += lipgloss.NewStyle().Foreground(colorGray).Render("Filter: "+m.filter+"  (/ edit • Esc clear)") + "\n"
		}
		title += "\n"
		vis := m.visibleModels()
		switch {
		case len(vis) > 0:
			names := make([]string, len(vis))
			pos := 0
			for k, idx := range vis {
				names[k] = m.models[idx]
				if idx == m.cursor {
					pos = k
				}
			}
			title += m.renderScrollableList(names, pos, m.height-9)
		case len(m.models) > 0:
			title += "(no model matches the filter)\n"
		case m.listError != "":
			title += lipgloss.NewStyle().Foreground(colorRed).Render("✗ "+m.listError) + "\n"
			if m.isCloud() {
				title += "Check your connection or key (Ctrl+K), then press Ctrl+R to try again.\n"
			} else {
				title += "Press Ctrl+R to try again (it starts Ollama if needed), or Ctrl+H to use another host.\n"
			}
		case m.isCloud():
			title += "(no models found)\n"
		default:
			title += "(no models installed - press Ctrl+P to pull one)\n"
		}
		if m.deleteTarget != "" {
			title += "\n" + lipgloss.NewStyle().Foreground(colorRed).Render(fmt.Sprintf("Delete %s from disk? (y/n)", m.deleteTarget)) + "\n"
		}
		if sl := m.statusLine(); sl != "" {
			title += "\n" + sl + "\n"
		}
		return m.pageStyle().Render(title)
	}

	// --- CHAT STATE UI ---
	modelName := m.models[m.cursor]
	header := m.headerView(modelName)

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
		statusText = lipgloss.NewStyle().Foreground(colorRed).Render(m.generatingStatus())
	} else if m.focusMode == "list" {
		listHelp := "[LIST] w/z/s Scroll • ↑/↓ Select • C Copy • N New • S Switch • P Settings • J Save • B Branch • Ctrl+O Hist"
		if m.supportsThinking {
			listHelp += " • Ctrl+T Think"
		}
		if m.supportsTools {
			listHelp += " • Ctrl+Y Tools"
		}
		statusText = lipgloss.NewStyle().Foreground(colorPurple).Render(listHelp)
	} else {
		chatHelp := "[CHAT MODE] Ctrl+F to attach file • Alt+Enter new line • Tab for list"
		if len(m.attachments) > 0 {
			chatHelp = lipgloss.NewStyle().Foreground(colorYellow).Bold(true).Render(fmt.Sprintf("[%d Attached] ", len(m.attachments))) + "[CHAT MODE] Ctrl+L to manage • Alt+Enter new line • Tab for list"
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
	hostFlag := flag.String("host", defaultHost, "Ollama server URL")
	consoleFlag := flag.Bool("console", false, "start in console mode (commands such as :status, :tokens, :ps)")
	flag.Parse()

	cfg := loadConfig()
	migrateLegacyChats()
	backend := cfg.Backend
	if backend != backendOpenRouter && backend != backendOpenAI {
		backend = backendOllama
	}

	host := normalizeHost(*hostFlag)
	if host == "" {
		host = defaultHost
	}

	// FIXED: only manage a local server, only start one if nothing is answering
	// yet, remember the process so "stop the server?" can really stop it, and
	// report a missing Ollama install instead of silently carrying on.
	var ollamaCmd *exec.Cmd
	ollamaMissing := false
	if backend == backendOllama && host == defaultHost && !ollamaReachable(host) {
		if path, err := exec.LookPath("ollama"); err == nil {
			cmd := exec.Command(path, "serve")
			if err := cmd.Start(); err == nil {
				ollamaCmd = cmd
				// Wait (up to ~5s) for it to come up instead of a blind sleep.
				for i := 0; i < 20 && !ollamaReachable(host); i++ {
					time.Sleep(250 * time.Millisecond)
				}
			}
		} else {
			ollamaMissing = true
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
	pi.SetHeight(4)
	pi.ShowLineNumbers = false

	var pInputs [2]textarea.Model
	pullPlaceholders := [2]string{"e.g. gemma3", "e.g. 4b (empty = latest)"}
	for i := range pInputs {
		t := textarea.New()
		t.Placeholder = pullPlaceholders[i]
		t.Prompt = "┃ "
		t.CharLimit = 64
		t.SetHeight(1)
		t.ShowLineNumbers = false
		pInputs[i] = t
	}
	var sInputs [4]textarea.Model
	placeholders := [4]string{"model default", "model default", "model default", "Ollama default"}
	for i := range sInputs {
		t := textarea.New()
		t.Placeholder = placeholders[i]
		t.Prompt = "┃ "
		t.CharLimit = 12
		t.SetHeight(1)
		t.ShowLineNumbers = false
		sInputs[i] = t
	}
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

	var kInputs [2]textinput.Model
	keyPlaceholders := [2]string{"sk-or-v1-...", "sk-..."}
	for i := range kInputs {
		kInputs[i] = newKeyInput(keyPlaceholders[i])
	}
	filterIn := textinput.New()
	filterIn.Prompt = "┃ "
	filterIn.Placeholder = "type to filter"
	filterIn.CharLimit = 64
	cvp := viewport.New(80, 15)
	cin := newConsoleInput()
	vp := viewport.New(80, 20)

	r, _ := glamour.NewTermRenderer(
		glamour.WithStylePath("dark"),
		glamour.WithWordWrap(70),
	)

	m := model{
		state:               stateLoading,
		textarea:            ta,
		promptInput:         pi,
		settingInputs:       sInputs,
		pullInputs:          pInputs,
		fileInput:           fi,
		hostInput:           hi,
		backend:             backend,
		keyInputs:           kInputs,
		filterInput:         filterIn,
		consoleVP:           cvp,
		consoleIn:           cin,
		snappedMsg:          -1,
		host:                host,
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
		ollamaCmd:           ollamaCmd,
		ollamaMissing:       ollamaMissing,
	}

	if *consoleFlag {
		m.openConsole()
	}
	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
