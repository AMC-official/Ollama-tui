package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// --- Cloud backends: OpenRouter and OpenAI (any OpenAI-compatible API) ---

const (
	backendOllama     = "ollama"
	backendOpenRouter = "openrouter"
	backendOpenAI     = "openai"
)

// cloudModel is what we know about one model of a cloud provider.
type cloudModel struct {
	ID       string
	Context  int
	Vision   bool
	Tools    bool
	Thinking bool
}

type cloudModelsMsg struct {
	provider string
	models   []cloudModel
}

func backendLabel(b string) string {
	switch b {
	case backendOpenRouter:
		return "OpenRouter"
	case backendOpenAI:
		return "OpenAI"
	}
	return "Ollama (local)"
}

func cloudBase(b string) string {
	switch b {
	case backendOpenRouter:
		return "https://openrouter.ai/api/v1"
	case backendOpenAI:
		return "https://api.openai.com/v1"
	}
	return ""
}

func (m model) isCloud() bool {
	return m.backend == backendOpenRouter || m.backend == backendOpenAI
}

// apiKey is the key of the current cloud backend: the saved one, or the
// OPENROUTER_API_KEY / OPENAI_API_KEY environment variable.
func (m model) apiKey() string {
	switch m.backend {
	case backendOpenRouter:
		if m.config.OpenRouterKey != "" {
			return m.config.OpenRouterKey
		}
		return strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	case backendOpenAI:
		if m.config.OpenAIKey != "" {
			return m.config.OpenAIKey
		}
		return strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	}
	return ""
}

// endpointLabel is the short server name shown in the header.
func (m model) endpointLabel() string {
	switch m.backend {
	case backendOpenRouter:
		return "openrouter.ai"
	case backendOpenAI:
		return "api.openai.com"
	}
	return strings.TrimPrefix(strings.TrimPrefix(m.host, "http://"), "https://")
}

func (m model) cloudCaps(id string) modelCapabilitiesMsg {
	if info, ok := m.cloudInfo[id]; ok {
		return modelCapabilitiesMsg{vision: info.Vision, thinking: info.Thinking, tools: info.Tools, maxCtx: info.Context}
	}
	return modelCapabilitiesMsg{}
}

// applyCaps stores what the selected model can do.
func (m *model) applyCaps(msg modelCapabilitiesMsg) {
	m.supportsVision = msg.vision
	m.supportsThinking = msg.thinking
	m.supportsTools = msg.tools
	m.useThinking = msg.thinking
	m.useTools = msg.tools
	m.modelMaxCtx = msg.maxCtx
	m.updateViewport()
}

// reloadModels loads the model list of the current backend again (and starts
// local Ollama first if it is not running).
func (m *model) reloadModels() tea.Cmd {
	m.listError = ""
	m.models = nil
	m.cloudInfo = nil
	m.cursor = 0
	m.state = stateLoading
	if m.isCloud() {
		return fetchCloudModels(m.backend, m.apiKey())
	}
	return startOllama(m.host)
}

// switchBackend moves on to the next backend and loads its models.
func (m *model) switchBackend() tea.Cmd {
	switch m.backend {
	case backendOpenRouter:
		m.backend = backendOpenAI
	case backendOpenAI:
		m.backend = backendOllama
	default:
		m.backend = backendOpenRouter
	}
	m.config.Backend = m.backend
	saveConfig(m.config)

	m.models = nil
	m.cloudInfo = nil
	m.cursor = 0
	m.filter = ""
	m.filterMode = false
	m.filterInput.SetValue("")
	m.listError = ""
	m.state = stateLoading
	if m.isCloud() {
		return fetchCloudModels(m.backend, m.apiKey())
	}
	// Ollama may not be running (we skip starting it when the app opens on a cloud backend).
	return startOllama(m.host)
}

func (m *model) localOnlyStatus() tea.Cmd {
	m.statusMsg = "That only works for local Ollama models (Ctrl+B switches backend)."
	m.statusIsError = true
	return clearStatusCmd(5 * time.Second)
}

// --- Model list filter ---

// visibleModels returns the indexes of the models that match the filter.
func (m model) visibleModels() []int {
	f := strings.ToLower(strings.TrimSpace(m.filter))
	idx := make([]int, 0, len(m.models))
	for i, name := range m.models {
		if f == "" || strings.Contains(strings.ToLower(name), f) {
			idx = append(idx, i)
		}
	}
	return idx
}

// moveCursor moves the selection through the filtered list. With delta 0 it
// only makes sure the selection is one of the visible models.
func (m *model) moveCursor(delta int) {
	vis := m.visibleModels()
	if len(vis) == 0 {
		return
	}
	pos := -1
	for k, i := range vis {
		if i == m.cursor {
			pos = k
			break
		}
	}
	if pos < 0 {
		m.cursor = vis[0]
		return
	}
	pos += delta
	if pos < 0 {
		pos = 0
	}
	if pos > len(vis)-1 {
		pos = len(vis) - 1
	}
	m.cursor = vis[pos]
}

// --- API keys screen ---

func newKeyInput(placeholder string) textinput.Model {
	ti := textinput.New()
	ti.Placeholder = placeholder
	ti.EchoMode = textinput.EchoPassword
	ti.EchoCharacter = '•'
	ti.CharLimit = 256
	ti.Prompt = "┃ "
	return ti
}

func (m *model) openKeys() {
	m.keyInputs[0].SetValue(m.config.OpenRouterKey)
	m.keyInputs[1].SetValue(m.config.OpenAIKey)
	m.keyFocus = 0
	m.focusKey()
	m.state = stateAPIKeys
}

func (m *model) focusKey() {
	for i := range m.keyInputs {
		m.keyInputs[i].Blur()
	}
	m.keyInputs[m.keyFocus].Focus()
}

func (m *model) closeKeys() {
	for i := range m.keyInputs {
		m.keyInputs[i].Blur()
	}
	m.state = stateSelectModel
}

func (m *model) saveKeys() tea.Cmd {
	m.config.OpenRouterKey = strings.TrimSpace(m.keyInputs[0].Value())
	m.config.OpenAIKey = strings.TrimSpace(m.keyInputs[1].Value())
	saveConfig(m.config)
	m.closeKeys()
	m.statusMsg = "API keys saved."
	m.statusIsError = false
	cmds := []tea.Cmd{clearStatusCmd(5 * time.Second)}
	if m.isCloud() {
		cmds = append(cmds, fetchCloudModels(m.backend, m.apiKey()))
	}
	return tea.Batch(cmds...)
}

func (m model) keysScreen() string {
	gray := lipgloss.NewStyle().Foreground(colorGray)
	label := func(i int, name string) string {
		if m.keyFocus == i {
			return aiLabelStyle.Render(name)
		}
		return gray.Render(name)
	}
	status := func(typed, env string) string {
		switch {
		case strings.TrimSpace(typed) != "":
			return lipgloss.NewStyle().Foreground(colorGreen).Render("✓ key set")
		case os.Getenv(env) != "":
			return lipgloss.NewStyle().Foreground(colorGreen).Render("✓ using $" + env)
		}
		return gray.Render("not set")
	}

	rows := []string{
		aiLabelStyle.Render("API keys"),
		"",
		label(0, "OpenRouter") + "  " + status(m.keyInputs[0].Value(), "OPENROUTER_API_KEY") + gray.Render("   get one at openrouter.ai/keys"),
		m.keyInputs[0].View(),
		"",
		label(1, "OpenAI") + "  " + status(m.keyInputs[1].Value(), "OPENAI_API_KEY") + gray.Render("   platform.openai.com/api-keys"),
		m.keyInputs[1].View(),
		"",
		gray.Render("Keys are saved as plain text in " + getConfigPath() + " (only you can read that file).\nYou can also set the OPENROUTER_API_KEY / OPENAI_API_KEY environment variables.\nLeave a field empty to remove its key.\n\nTab: switch field • Enter: next field (saves on the last) • Ctrl+S: save • Esc: cancel"),
	}
	if sl := m.statusLine(); sl != "" {
		rows = append(rows, "", sl)
	}
	return m.pageStyle().Render(lipgloss.JoinVertical(lipgloss.Left, rows...))
}

// --- Loading the model lists ---

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// isOpenAIChatModel keeps the chat models and drops embeddings, audio, images...
func isOpenAIChatModel(id string) bool {
	id = strings.ToLower(id)
	okPrefix := strings.HasPrefix(id, "gpt-") || strings.HasPrefix(id, "chatgpt-") ||
		strings.HasPrefix(id, "o1") || strings.HasPrefix(id, "o3") || strings.HasPrefix(id, "o4")
	if !okPrefix {
		return false
	}
	for _, bad := range []string{"realtime", "audio", "transcribe", "tts", "image", "instruct", "embedding", "search", "moderation"} {
		if strings.Contains(id, bad) {
			return false
		}
	}
	return true
}

// openAICaps guesses what an OpenAI model can do from its name (the API does not say).
func openAICaps(id string) (vision, tools, thinking bool) {
	id = strings.ToLower(id)
	thinking = strings.HasPrefix(id, "o1") || strings.HasPrefix(id, "o3") || strings.HasPrefix(id, "o4") || strings.Contains(id, "gpt-5")
	vision = strings.Contains(id, "gpt-4o") || strings.Contains(id, "gpt-4.1") || strings.Contains(id, "gpt-4-turbo") ||
		strings.Contains(id, "gpt-5") || strings.HasPrefix(id, "o3") || strings.HasPrefix(id, "o4") || id == "o1"
	tools = !strings.Contains(id, "o1-mini") && !strings.Contains(id, "o1-preview")
	return vision, tools, thinking
}

// parseCloudModels reads the reply of GET /models.
func parseCloudModels(provider string, data []byte) ([]cloudModel, error) {
	var res struct {
		Data []struct {
			ID            string `json:"id"`
			ContextLength int    `json:"context_length"`
			Architecture  struct {
				InputModalities []string `json:"input_modalities"`
			} `json:"architecture"`
			SupportedParameters []string `json:"supported_parameters"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &res); err != nil {
		return nil, err
	}
	var out []cloudModel
	for _, d := range res.Data {
		if d.ID == "" {
			continue
		}
		cm := cloudModel{ID: d.ID, Context: d.ContextLength}
		if provider == backendOpenAI {
			if !isOpenAIChatModel(d.ID) {
				continue
			}
			cm.Vision, cm.Tools, cm.Thinking = openAICaps(d.ID)
		} else {
			cm.Vision = containsString(d.Architecture.InputModalities, "image")
			cm.Tools = containsString(d.SupportedParameters, "tools")
			cm.Thinking = containsString(d.SupportedParameters, "reasoning") || containsString(d.SupportedParameters, "include_reasoning")
		}
		out = append(out, cm)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func fetchCloudModels(provider, key string) tea.Cmd {
	return func() tea.Msg {
		label := backendLabel(provider)
		if provider == backendOpenAI && key == "" {
			return errMsg{fmt.Errorf("add your OpenAI API key first (Ctrl+K)")}
		}
		req, err := http.NewRequest("GET", cloudBase(provider)+"/models", nil)
		if err != nil {
			return errMsg{err}
		}
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := apiClient.Do(req)
		if err != nil {
			return errMsg{fmt.Errorf("could not reach %s: %v", label, err)}
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
		if err != nil {
			return errMsg{err}
		}
		if resp.StatusCode != http.StatusOK {
			return errMsg{fmt.Errorf("%s answered %s: %s", label, resp.Status, apiErrorDetail(body))}
		}
		models, err := parseCloudModels(provider, body)
		if err != nil {
			return errMsg{fmt.Errorf("unexpected reply from %s: %v", label, err)}
		}
		return cloudModelsMsg{provider: provider, models: models}
	}
}

// --- Talking to an OpenAI-compatible API ---

type oaiImageURL struct {
	URL string `json:"url"`
}
type oaiPart struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *oaiImageURL `json:"image_url,omitempty"`
}
type oaiToolFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // a JSON document inside a string
}
type oaiToolCall struct {
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Function oaiToolFunc `json:"function"`
}
type oaiMsg struct {
	Role       string        `json:"role"`
	Content    interface{}   `json:"content"` // text, a list of parts (images), or null
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// imageDataURL turns base64 image data into a data: URL, guessing the type from the bytes.
func imageDataURL(b64 string) string {
	head := b64
	if len(head) > 400 {
		head = head[:400]
	}
	data, _ := base64.StdEncoding.DecodeString(head)
	mime := http.DetectContentType(data)
	if !strings.HasPrefix(mime, "image/") {
		mime = "image/png"
	}
	return "data:" + mime + ";base64," + b64
}

// toOpenAIMessages converts our message list to the OpenAI format: images
// become image_url parts, tool calls carry an id and JSON-string arguments,
// and every tool result points at the call it answers.
func toOpenAIMessages(msgs []OllamaChatMsg) []oaiMsg {
	type pendingCall struct{ id, name string }
	out := make([]oaiMsg, 0, len(msgs))
	var pending []pendingCall
	generated := 0

	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			om := oaiMsg{Role: "assistant"}
			if m.Content != "" {
				om.Content = m.Content
			}
			pending = pending[:0]
			for _, c := range m.ToolCalls {
				id := c.ID
				if id == "" { // history from a local model has no ids: make some up
					id = fmt.Sprintf("call_%d", generated)
					generated++
				}
				args, _ := json.Marshal(c.Function.Arguments)
				if len(c.Function.Arguments) == 0 {
					args = []byte("{}")
				}
				om.ToolCalls = append(om.ToolCalls, oaiToolCall{
					ID:       id,
					Type:     "function",
					Function: oaiToolFunc{Name: c.Function.Name, Arguments: string(args)},
				})
				pending = append(pending, pendingCall{id: id, name: c.Function.Name})
			}
			out = append(out, om)

		case "tool":
			id := m.ToolCallID
			if id == "" {
				for k, p := range pending {
					if p.name == m.ToolName {
						id = p.id
						pending = append(pending[:k], pending[k+1:]...)
						break
					}
				}
			}
			out = append(out, oaiMsg{Role: "tool", ToolCallID: id, Content: m.Content})

		case "user":
			if len(m.Images) == 0 {
				out = append(out, oaiMsg{Role: "user", Content: m.Content})
				continue
			}
			parts := []oaiPart{{Type: "text", Text: m.Content}}
			for _, img := range m.Images {
				parts = append(parts, oaiPart{Type: "image_url", ImageURL: &oaiImageURL{URL: imageDataURL(img)}})
			}
			out = append(out, oaiMsg{Role: "user", Content: parts})

		default: // system
			out = append(out, oaiMsg{Role: m.Role, Content: m.Content})
		}
	}
	return out
}

// errorText pulls the message out of an API "error" value, which is either
// {"message": "..."} or a plain string.
func errorText(raw json.RawMessage) string {
	var obj struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.Message != "" {
		return obj.Message
	}
	var s string
	if json.Unmarshal(raw, &s) == nil && s != "" {
		return s
	}
	return string(raw)
}

// apiErrorDetail finds the reason in an error reply body.
func apiErrorDetail(body []byte) string {
	var e struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && len(e.Error) > 0 && string(e.Error) != "null" {
		return errorText(e.Error)
	}
	return strings.TrimSpace(string(body))
}

type pendingToolCall struct {
	id   string
	name string
	args strings.Builder
}

// streamOpenAI reads a server-sent-events stream and hands text, reasoning and
// finished tool calls to send. It stops early if send returns false.
func streamOpenAI(r io.Reader, send func(OllamaGenerateRes) bool) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	calls := map[int]*pendingToolCall{}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue // blank lines and ": keep-alive" comments
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					Reasoning        string `json:"reasoning"`
					ReasoningContent string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Error json.RawMessage `json:"error"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if len(chunk.Error) > 0 && string(chunk.Error) != "null" {
			return fmt.Errorf("%s", errorText(chunk.Error))
		}

		for _, c := range chunk.Choices {
			d := c.Delta
			think := d.Reasoning + d.ReasoningContent
			if d.Content != "" || think != "" {
				if !send(OllamaGenerateRes{Response: d.Content, Thinking: think}) {
					return nil
				}
			}
			// Tool calls arrive in pieces: collect them by index.
			for _, tc := range d.ToolCalls {
				pc := calls[tc.Index]
				if pc == nil {
					pc = &pendingToolCall{}
					calls[tc.Index] = pc
				}
				if tc.ID != "" {
					pc.id = tc.ID
				}
				if tc.Function.Name != "" {
					pc.name = tc.Function.Name
				}
				pc.args.WriteString(tc.Function.Arguments)
			}
		}
		if chunk.Usage != nil && (chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0) {
			if !send(OllamaGenerateRes{Usage: &UsageInfo{Prompt: chunk.Usage.PromptTokens, Completion: chunk.Usage.CompletionTokens}}) {
				return nil
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	if len(calls) > 0 {
		order := make([]int, 0, len(calls))
		for i := range calls {
			order = append(order, i)
		}
		sort.Ints(order)
		var out []ToolCall
		for _, i := range order {
			pc := calls[i]
			args := map[string]interface{}{}
			if s := strings.TrimSpace(pc.args.String()); s != "" {
				json.Unmarshal([]byte(s), &args)
			}
			out = append(out, ToolCall{ID: pc.id, Function: ToolCallFunction{Name: pc.name, Arguments: args}})
		}
		send(OllamaGenerateRes{ToolCalls: out})
	}
	return nil
}

// startGenerateCloud is startGenerate for OpenRouter / OpenAI: same channel,
// same kind of results, so the rest of the app does not care which one it is.
func startGenerateCloud(ctx context.Context, provider, key, modelID string, msgs []OllamaChatMsg, options map[string]interface{}, tools []ToolDef, ch chan OllamaGenerateRes) tea.Cmd {
	return func() tea.Msg {
		defer close(ch)

		send := func(res OllamaGenerateRes) bool {
			select {
			case ch <- res:
				return true
			case <-ctx.Done():
				return false
			}
		}

		label := backendLabel(provider)
		if key == "" {
			send(OllamaGenerateRes{Error: "No " + label + " API key set. Press Ctrl+K on the model screen to add one."})
			return nil
		}

		body := map[string]interface{}{
			"model":    modelID,
			"messages": toOpenAIMessages(msgs),
			"stream":   true,
		}
		for _, k := range []string{"temperature", "top_p"} {
			if v, ok := options[k]; ok {
				body[k] = v
			}
		}
		if v, ok := options["top_k"]; ok && provider == backendOpenRouter {
			body["top_k"] = v // OpenAI itself does not accept top_k
		}
		if len(tools) > 0 {
			body["tools"] = tools
		}
		if provider == backendOpenAI {
			body["stream_options"] = map[string]interface{}{"include_usage": true}
		}
		payload, err := json.Marshal(body)
		if err != nil {
			send(OllamaGenerateRes{Error: err.Error()})
			return nil
		}

		req, err := http.NewRequestWithContext(ctx, "POST", cloudBase(provider)+"/chat/completions", bytes.NewReader(payload))
		if err != nil {
			send(OllamaGenerateRes{Error: err.Error()})
			return nil
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Accept", "text/event-stream")
		if provider == backendOpenRouter {
			req.Header.Set("HTTP-Referer", "https://github.com/AMC-official/Ollama-tui")
			req.Header.Set("X-Title", "ollama-tui")
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			if ctx.Err() == nil {
				send(OllamaGenerateRes{Error: "failed to reach " + label + ": " + err.Error()})
			}
			return nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
			send(OllamaGenerateRes{Error: fmt.Sprintf("%s answered %s: %s", label, resp.Status, apiErrorDetail(b))})
			return nil
		}

		if err := streamOpenAI(resp.Body, send); err != nil && ctx.Err() == nil {
			send(OllamaGenerateRes{Error: err.Error()})
		}
		return nil
	}
}
