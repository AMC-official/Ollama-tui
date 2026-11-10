package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// --- Replies and tool calls ---

// beginReply starts streaming an answer to the conversation so far. It serves
// both a normal message (fresh) and the follow-up after tool results came back.
func (m *model) beginReply(images []string, fresh, withTools bool) tea.Cmd {
	// Built here, on the UI goroutine, so the streaming goroutine never touches m.messages.
	chatMsgs := buildChatMessages(m.messages, m.systemPrompt, images)
	m.messages = append(m.messages, ChatMessage{Role: "AI"})

	var thinkPtr *bool
	if m.supportsThinking {
		think := m.useThinking
		thinkPtr = &think
	}
	var defs []ToolDef
	if withTools && m.supportsTools {
		defs = toolDefs()
	}

	m.selectedMsg = -1
	m.isThinking = true
	m.gotOutput = false
	m.autoScroll = true
	if fresh {
		m.genStart = time.Now()
		m.toolRounds = 0
	}

	ctx, cancel := context.WithCancel(context.Background())
	m.cancelStream = cancel
	ch := make(chan OllamaGenerateRes)
	m.streamChan = ch
	m.pendingCalls = nil

	m.liveBuilder.Reset()
	m.liveThinkingBuilder.Reset()
	m.updateViewport()
	m.viewport.GotoBottom()

	var gen tea.Cmd
	if m.isCloud() {
		gen = startGenerateCloud(ctx, m.backend, m.apiKey(), m.models[m.cursor], chatMsgs, buildOptions(m.config, 0), defs, ch)
	} else {
		gen = startGenerate(ctx, m.host, m.models[m.cursor], chatMsgs, thinkPtr, buildOptions(m.config, m.numCtxSetting()), defs, ch)
	}
	return tea.Batch(gen, waitForStream(ch), tickCmd(ch), renderTickCmd(ch))
}

// runPendingTools runs the tools the model asked for, adds their results to
// the chat, and starts the model's follow-up answer.
func (m *model) runPendingTools() tea.Cmd {
	calls := m.pendingCalls
	m.pendingCalls = nil

	if last := len(m.messages) - 1; last >= 0 && m.messages[last].Role == "AI" {
		m.messages[last].ToolCalls = calls
	}
	for _, c := range calls {
		m.messages = append(m.messages, ChatMessage{
			Role:     "Tool",
			ToolName: c.Function.Name,
			ToolCallID: c.ID,
			Content:  runTool(c.Function.Name, c.Function.Arguments),
		})
	}

	// After a few rounds the model gets no tools any more, so it has to answer.
	withTools := m.toolRounds < maxToolRounds
	m.toolRounds++
	return m.beginReply(nil, false, withTools)
}

// shorten squeezes text onto one line and cuts it to at most max characters.
func shorten(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "…"
}

func describeCalls(calls []ToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, c := range calls {
		args := ""
		if len(c.Function.Arguments) > 0 {
			b, _ := json.Marshal(c.Function.Arguments)
			args = shorten(string(b), 80)
		}
		parts = append(parts, c.Function.Name+"("+args+")")
	}
	return strings.Join(parts, ", ")
}

// --- The tools themselves (all local and read-only) ---

type toolSpec struct {
	name        string
	description string
	properties  map[string]interface{}
	required    []string
	run         func(args map[string]interface{}) (string, error)
}

func strProp(desc string) map[string]interface{} {
	return map[string]interface{}{"type": "string", "description": desc}
}

var toolSpecs = []toolSpec{
	{
		name:        "get_current_time",
		description: "Get the current date and time on the user's computer. Leave timezone empty for the user's local time; only set it when the user asks about another place.",
		properties:  map[string]interface{}{"timezone": strProp("Only for another place than the user's own: an IANA time zone name such as Europe/Paris or America/New_York.")},
		run:         toolCurrentTime,
	},
	{
		name:        "calculate",
		description: "Evaluate a math expression exactly instead of guessing. Supports + - * / % ^, parentheses, pi, e and the functions sqrt, abs, round, floor, ceil.",
		properties:  map[string]interface{}{"expression": strProp("The expression, for example (12.5 * 4) / 3.")},
		required:    []string{"expression"},
		run:         toolCalculate,
	},
	{
		name:        "list_files",
		description: "List the files and folders in a folder, relative to the folder the user started this app in. Hidden files are not shown.",
		properties:  map[string]interface{}{"path": strProp("Folder to list. Defaults to the current folder.")},
		run:         toolListFiles,
	},
	{
		name:        "read_file",
		description: "Read a text file (the first 20000 characters), relative to the folder the user started this app in.",
		properties:  map[string]interface{}{"path": strProp("Path of the file to read.")},
		required:    []string{"path"},
		run:         toolReadFile,
	},
}

// toolDefs describes every tool to the model.
func toolDefs() []ToolDef {
	defs := make([]ToolDef, 0, len(toolSpecs))
	for _, t := range toolSpecs {
		required := t.required
		if required == nil {
			required = []string{}
		}
		defs = append(defs, ToolDef{
			Type: "function",
			Function: ToolDefFunction{
				Name:        t.name,
				Description: t.description,
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": t.properties,
					"required":   required,
				},
			},
		})
	}
	return defs
}

// runTool runs one tool and returns its result as text. Errors come back as
// text too, so the model can read them and try something else.
func runTool(name string, args map[string]interface{}) string {
	for _, t := range toolSpecs {
		if t.name == name {
			out, err := t.run(args)
			if err != nil {
				return "error: " + err.Error()
			}
			return out
		}
	}
	return "error: unknown tool " + name
}

func argString(args map[string]interface{}, key string) string {
	switch v := args[key].(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(v)
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func toolCurrentTime(args map[string]interface{}) (string, error) {
	now := time.Now()
	if tz := argString(args, "timezone"); tz != "" {
		loc, err := time.LoadLocation(tz)
		if err != nil {
			return "", fmt.Errorf("unknown time zone %q (use a name like Europe/Paris)", tz)
		}
		now = now.In(loc)
	}
	return now.Format("Monday, 2 January 2006, 15:04:05 MST (UTC-07:00)"), nil
}

func toolCalculate(args map[string]interface{}) (string, error) {
	expr := argString(args, "expression")
	if expr == "" {
		return "", fmt.Errorf("missing expression")
	}
	v, err := evalExpr(expr)
	if err != nil {
		return "", err
	}
	return strconv.FormatFloat(v, 'g', 12, 64), nil
}

// safePath resolves p relative to the folder the app was started in and
// refuses anything outside of it (also through symlinks) and hidden files.
func safePath(p string) (string, error) {
	root, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if p == "" {
		p = "."
	}
	full := p
	if !filepath.IsAbs(full) {
		full = filepath.Join(root, full)
	}
	realRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	realFull, err := filepath.EvalSymlinks(filepath.Clean(full))
	if err != nil {
		return "", fmt.Errorf("cannot access %q", p)
	}
	rel, err := filepath.Rel(realRoot, realFull)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("%q is outside the folder the app was started in", p)
	}
	for _, part := range strings.Split(rel, string(filepath.Separator)) {
		if part != "." && strings.HasPrefix(part, ".") {
			return "", fmt.Errorf("hidden files and folders are not accessible")
		}
	}
	return realFull, nil
}

func toolListFiles(args map[string]interface{}) (string, error) {
	dir, err := safePath(argString(args, "path"))
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var names []string
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if e.IsDir() {
			name += "/"
		}
		names = append(names, name)
		if len(names) == 200 {
			names = append(names, "... (more files not shown)")
			break
		}
	}
	if len(names) == 0 {
		return "(empty folder)", nil
	}
	return strings.Join(names, "\n"), nil
}

func toolReadFile(args map[string]interface{}) (string, error) {
	path, err := safePath(argString(args, "path"))
	if err != nil {
		return "", err
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("that is a folder, use list_files instead")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	data, err := io.ReadAll(io.LimitReader(f, int64(maxAttachChars*4)))
	if err != nil {
		return "", err
	}
	head := data
	if len(head) > 1024 {
		head = head[:1024]
	}
	if bytes.IndexByte(head, 0) >= 0 {
		return "", fmt.Errorf("that looks like a binary file, not text")
	}
	text := string(data)
	if len(text) > maxAttachChars {
		text = strings.ToValidUTF8(text[:maxAttachChars], "") + "\n...[TRUNCATED]..."
	}
	return text, nil
}

// --- A small, safe calculator (no eval, no dependencies) ---

type exprParser struct {
	s   string
	pos int
}

// evalExpr evaluates + - * / % ^, parentheses, unary minus, pi, e and a few functions.
func evalExpr(s string) (float64, error) {
	p := &exprParser{s: s}
	v, err := p.expr()
	if err != nil {
		return 0, err
	}
	if p.peek() != 0 {
		return 0, fmt.Errorf("unexpected %q at position %d", p.s[p.pos], p.pos+1)
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("the result is not a finite number")
	}
	return v, nil
}

// peek returns the next byte that is not a space (0 at the end).
func (p *exprParser) peek() byte {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t') {
		p.pos++
	}
	if p.pos < len(p.s) {
		return p.s[p.pos]
	}
	return 0
}

func (p *exprParser) expr() (float64, error) {
	left, err := p.term()
	if err != nil {
		return 0, err
	}
	for {
		op := p.peek()
		if op != '+' && op != '-' {
			return left, nil
		}
		p.pos++
		right, err := p.term()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
}

func (p *exprParser) term() (float64, error) {
	left, err := p.unary()
	if err != nil {
		return 0, err
	}
	for {
		op := p.peek()
		if op != '*' && op != '/' && op != '%' {
			return left, nil
		}
		p.pos++
		right, err := p.unary()
		if err != nil {
			return 0, err
		}
		switch op {
		case '*':
			left *= right
		case '/':
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		case '%':
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left = math.Mod(left, right)
		}
	}
}

func (p *exprParser) unary() (float64, error) {
	switch p.peek() {
	case '-':
		p.pos++
		v, err := p.unary()
		return -v, err
	case '+':
		p.pos++
		return p.unary()
	}
	return p.power()
}

// power is right-associative (2^3^2 = 2^9) and binds tighter than unary minus (-2^2 = -4).
func (p *exprParser) power() (float64, error) {
	base, err := p.primary()
	if err != nil {
		return 0, err
	}
	if p.peek() == '^' {
		p.pos++
		exp, err := p.unary()
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

func isLetter(c byte) bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }
func isDigit(c byte) bool  { return c >= '0' && c <= '9' }

func (p *exprParser) primary() (float64, error) {
	c := p.peek()
	switch {
	case c == 0:
		return 0, fmt.Errorf("the expression ends too early")

	case c == '(':
		p.pos++
		v, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		p.pos++
		return v, nil

	case isDigit(c) || c == '.':
		start := p.pos
		for p.pos < len(p.s) && (isDigit(p.s[p.pos]) || p.s[p.pos] == '.') {
			p.pos++
		}
		v, err := strconv.ParseFloat(p.s[start:p.pos], 64)
		if err != nil {
			return 0, fmt.Errorf("bad number %q", p.s[start:p.pos])
		}
		return v, nil

	case isLetter(c):
		start := p.pos
		for p.pos < len(p.s) && isLetter(p.s[p.pos]) {
			p.pos++
		}
		name := strings.ToLower(p.s[start:p.pos])
		switch name {
		case "pi":
			return math.Pi, nil
		case "e":
			return math.E, nil
		}
		if p.peek() != '(' {
			return 0, fmt.Errorf("unknown name %q", name)
		}
		p.pos++
		arg, err := p.expr()
		if err != nil {
			return 0, err
		}
		if p.peek() != ')' {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		p.pos++
		switch name {
		case "sqrt":
			if arg < 0 {
				return 0, fmt.Errorf("sqrt of a negative number")
			}
			return math.Sqrt(arg), nil
		case "abs":
			return math.Abs(arg), nil
		case "round":
			return math.Round(arg), nil
		case "floor":
			return math.Floor(arg), nil
		case "ceil":
			return math.Ceil(arg), nil
		}
		return 0, fmt.Errorf("unknown function %q", name)
	}
	return 0, fmt.Errorf("unexpected %q at position %d", c, p.pos+1)
}
