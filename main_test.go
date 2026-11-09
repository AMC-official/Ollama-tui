package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/charmbracelet/bubbles/viewport"
	"github.com/charmbracelet/lipgloss"
)

func testFloat(v float64) *float64 { return &v }
func testInt(v int) *int           { return &v }

// --- buildChatMessages ---

func TestBuildChatMessages(t *testing.T) {
	attached := "Context from attached files:\n--- a.txt ---\nhello\n\n"

	tests := []struct {
		name    string
		history []ChatMessage
		system  string
		images  []string
		want    []OllamaChatMsg
	}{
		{
			name:    "system prompt goes first",
			history: []ChatMessage{{Role: "You", Content: "hi"}},
			system:  "be nice",
			want: []OllamaChatMsg{
				{Role: "system", Content: "be nice"},
				{Role: "user", Content: "hi"},
			},
		},
		{
			name:    "blank system prompt is ignored",
			history: []ChatMessage{{Role: "You", Content: "hi"}},
			system:  "   ",
			want:    []OllamaChatMsg{{Role: "user", Content: "hi"}},
		},
		{
			name: "UI-only system notes are not sent",
			history: []ChatMessage{
				{Role: "System", Content: "Loaded model: gemma3:4b"},
				{Role: "You", Content: "hi"},
				{Role: "System", Content: "📎 Attached: a.txt"},
			},
			want: []OllamaChatMsg{{Role: "user", Content: "hi"}},
		},
		{
			name:    "attached file text goes before the prompt",
			history: []ChatMessage{{Role: "You", Content: "what is this?", Context: attached}},
			want:    []OllamaChatMsg{{Role: "user", Content: attached + "User Prompt: what is this?"}},
		},
		{
			name: "the stopped-by-user marker is removed",
			history: []ChatMessage{
				{Role: "You", Content: "tell me a story"},
				{Role: "AI", Content: "Once upon a time [Stopped by user]"},
			},
			want: []OllamaChatMsg{
				{Role: "user", Content: "tell me a story"},
				{Role: "assistant", Content: "Once upon a time"},
			},
		},
		{
			name: "empty AI replies are skipped",
			history: []ChatMessage{
				{Role: "You", Content: "a"},
				{Role: "AI", Content: ""},
				{Role: "AI", Content: "   "},
				{Role: "You", Content: "b"},
			},
			want: []OllamaChatMsg{
				{Role: "user", Content: "a"},
				{Role: "user", Content: "b"},
			},
		},
		{
			name: "images are attached to the last user message only",
			history: []ChatMessage{
				{Role: "You", Content: "one"},
				{Role: "AI", Content: "two"},
				{Role: "You", Content: "three"},
			},
			images: []string{"IMG"},
			want: []OllamaChatMsg{
				{Role: "user", Content: "one"},
				{Role: "assistant", Content: "two"},
				{Role: "user", Content: "three", Images: []string{"IMG"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildChatMessages(tt.history, tt.system, tt.images)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got  %+v\nwant %+v", got, tt.want)
			}
		})
	}
}

// --- settings / options ---

func TestBuildOptions(t *testing.T) {
	if got := buildOptions(Config{}, 0); got != nil {
		t.Fatalf("expected nil when nothing is set, got %v", got)
	}

	cfg := Config{Temperature: testFloat(0.7), TopP: testFloat(0.9), TopK: testInt(40)}
	want := map[string]interface{}{"temperature": 0.7, "top_p": 0.9, "top_k": 40, "num_ctx": 8192}
	if got := buildOptions(cfg, 8192); !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}

	// A temperature of 0 is a real choice and must still be sent.
	zero := buildOptions(Config{Temperature: testFloat(0)}, 0)
	if v, ok := zero["temperature"]; !ok || v != 0.0 {
		t.Errorf("temperature 0 should be sent, got %v", zero)
	}
}

func TestContextSettings(t *testing.T) {
	m := model{config: Config{NumCtx: 8192}, modelMaxCtx: 4096}
	if got := m.numCtxSetting(); got != 4096 {
		t.Errorf("setting should be capped to the model's max: got %d, want 4096", got)
	}

	m.modelMaxCtx = 0 // unknown maximum: use the setting as is
	if got := m.numCtxSetting(); got != 8192 {
		t.Errorf("got %d, want 8192", got)
	}
	if got := m.effectiveCtx(); got != 8192 {
		t.Errorf("effectiveCtx: got %d, want 8192", got)
	}

	m = model{}
	if got := m.numCtxSetting(); got != 0 {
		t.Errorf("unset setting should be 0, got %d", got)
	}
	if got := m.effectiveCtx(); got != contextWindowTokens {
		t.Errorf("unset setting should fall back to %d, got %d", contextWindowTokens, got)
	}
}

func TestFormatFloat(t *testing.T) {
	if got := formatFloat(nil); got != "" {
		t.Errorf("nil should give an empty string, got %q", got)
	}
	if got := formatFloat(testFloat(0.7)); got != "0.7" {
		t.Errorf("got %q, want 0.7", got)
	}
	if got := formatFloat(testFloat(2)); got != "2" {
		t.Errorf("got %q, want 2", got)
	}
}

func TestContextLengthFromInfo(t *testing.T) {
	tests := []struct {
		name string
		info map[string]interface{}
		want int
	}{
		{"architecture key", map[string]interface{}{"general.architecture": "llama", "llama.context_length": float64(131072)}, 131072},
		{"fallback to any context_length key", map[string]interface{}{"qwen3vl.context_length": float64(262144)}, 262144},
		{"missing", map[string]interface{}{"general.architecture": "llama"}, 0},
		{"nil map", nil, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := contextLengthFromInfo(tt.info); got != tt.want {
				t.Errorf("got %d, want %d", got, tt.want)
			}
		})
	}
}

// --- text and formatting helpers ---

func TestNormalizeText(t *testing.T) {
	if got := normalizeText("hello   world\n\n  foo\t bar "); got != "hello world foo bar" {
		t.Errorf("whitespace should collapse, got %q", got)
	}

	// Too long: cut and marked.
	suffix := "\n...[TRUNCATED]..."
	long := normalizeText(strings.Repeat("a", 25000))
	if len(long) != maxAttachChars+len(suffix) || !strings.HasSuffix(long, suffix) {
		t.Errorf("long text should be cut to %d chars plus the marker, got length %d", maxAttachChars, len(long))
	}

	// Cutting in the middle of a multi-byte character must not leave broken UTF-8.
	euros := normalizeText(strings.Repeat("€", 8000)) // 3 bytes each, 20000 is not a multiple of 3
	if !utf8.ValidString(euros) {
		t.Error("truncated text is not valid UTF-8")
	}
}

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		in   time.Duration
		want string
	}{
		{0, "0.0s"},
		{3400 * time.Millisecond, "3.4s"},
		{10 * time.Second, "10s"},
		{42 * time.Second, "42s"},
		{60 * time.Second, "1m 0s"},
		{125 * time.Second, "2m 5s"},
		{3725 * time.Second, "1h 2m"},
	}
	for _, tt := range tests {
		if got := formatDuration(tt.in); got != tt.want {
			t.Errorf("formatDuration(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestHumanBytes(t *testing.T) {
	tests := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{999, "999 B"},
		{1000, "1.0 kB"},
		{1500, "1.5 kB"},
		{291_600_000, "291.6 MB"},
		{3_300_000_000, "3.3 GB"},
	}
	for _, tt := range tests {
		if got := humanBytes(tt.in); got != tt.want {
			t.Errorf("humanBytes(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestProgressBar(t *testing.T) {
	bar := progressBar(0.5, 10)
	if strings.Count(bar, "█") != 5 || strings.Count(bar, "░") != 5 {
		t.Errorf("half bar should have 5 full and 5 empty cells, got %q", bar)
	}
	if bar := progressBar(2, 4); strings.Count(bar, "█") != 4 || strings.Count(bar, "░") != 0 {
		t.Errorf("values above 1 should clamp to a full bar, got %q", bar)
	}
	if bar := progressBar(-1, 4); strings.Count(bar, "█") != 0 || strings.Count(bar, "░") != 4 {
		t.Errorf("negative values should clamp to an empty bar, got %q", bar)
	}
}

func TestFriendlyPullStatus(t *testing.T) {
	tests := map[string]string{
		"pulling manifest":        "Fetching the model manifest...",
		"pulling 6a0746a1ec1a":    "Downloading...",
		"verifying sha256 digest": "Verifying the download...",
		"writing manifest":        "Writing the manifest...",
		"success":                 "Done!",
		"something else":          "something else",
	}
	for in, want := range tests {
		if got := friendlyPullStatus(in); got != want {
			t.Errorf("friendlyPullStatus(%q) = %q, want %q", in, got, want)
		}
	}
}

// --- model names and hosts ---

func TestValidModelPart(t *testing.T) {
	tests := []struct {
		in         string
		allowSlash bool
		want       bool
	}{
		{"gemma3", false, true},
		{"qwen3-vl", false, true},
		{"llama3.1", false, true},
		{"4b", false, true},
		{"", false, false},
		{"Gemma", false, false}, // upper case is not allowed
		{"-bad", false, false},  // must start with a letter or digit
		{"a b", false, false},
		{"../x", true, false},
		{"user/model", true, true},
		{"user/model", false, false},
	}
	for _, tt := range tests {
		if got := validModelPart(tt.in, tt.allowSlash); got != tt.want {
			t.Errorf("validModelPart(%q, %v) = %v, want %v", tt.in, tt.allowSlash, got, tt.want)
		}
	}
}

func TestNormalizeHost(t *testing.T) {
	tests := map[string]string{
		"localhost:11434":           "http://localhost:11434",
		"http://192.168.1.5:11434/": "http://192.168.1.5:11434",
		"  https://example.com/ ":   "https://example.com",
		"":                          "",
	}
	for in, want := range tests {
		if got := normalizeHost(in); got != want {
			t.Errorf("normalizeHost(%q) = %q, want %q", in, got, want)
		}
	}
}

// --- attaching files ---

func TestProcessFileText(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notes.txt")
	if err := os.WriteFile(path, []byte("hello   world\n\n  foo"), 0o644); err != nil {
		t.Fatal(err)
	}

	// A path wrapped in quotes (as terminals paste it) must work too.
	msg, ok := processFile(`"` + path + `"`).(fileAttachedMsg)
	if !ok {
		t.Fatal("expected a fileAttachedMsg")
	}
	a := msg.attachment
	if a.Name != "notes.txt" || a.Text != "hello world foo" || a.IsImage {
		t.Errorf("unexpected attachment: %+v", a)
	}
}

func TestProcessFileImage(t *testing.T) {
	data := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	path := filepath.Join(t.TempDir(), "pic.png")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	msg, ok := processFile(path).(fileAttachedMsg)
	if !ok {
		t.Fatal("expected a fileAttachedMsg")
	}
	a := msg.attachment
	if !a.IsImage || a.Text != "[Image Data]" || a.Base64 != base64.StdEncoding.EncodeToString(data) {
		t.Errorf("unexpected image attachment: %+v", a)
	}
}

func TestProcessFileErrors(t *testing.T) {
	if msg, ok := processFile("program.exe").(fileErrorMsg); !ok || !strings.Contains(msg.err, ".exe") {
		t.Errorf("unsupported types should give a fileErrorMsg naming the extension, got %v", msg)
	}
	if _, ok := processFile(filepath.Join(t.TempDir(), "missing.txt")).(fileErrorMsg); !ok {
		t.Error("a missing file should give a fileErrorMsg")
	}
}

// --- header ---

func TestTokenMeter(t *testing.T) {
	tests := []struct {
		name  string
		chars int
		color lipgloss.Color
	}{
		{"low usage is green", 400, colorGreen},
		{"above 75% is yellow", 4096 * 4 * 8 / 10, colorYellow},
		{"above 90% is red", 4096 * 4, colorRed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{messages: []ChatMessage{{Role: "You", Content: strings.Repeat("a", tt.chars)}}}
			used, _, color := m.tokenMeter()
			if used != tt.chars/4 {
				t.Errorf("used = %d, want %d", used, tt.chars/4)
			}
			if color != tt.color {
				t.Errorf("color = %v, want %v", color, tt.color)
			}
		})
	}
}

func TestHeaderLines(t *testing.T) {
	tests := []struct {
		name          string
		width, height int
		want          int
	}{
		{"big window has the two-line header", 100, 40, 3},
		{"narrow window is compact", 50, 40, 2},
		{"short window is compact", 100, 20, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{width: tt.width, height: tt.height}
			if got := m.headerLines(); got != tt.want {
				t.Errorf("headerLines() = %d, want %d", got, tt.want)
			}
		})
	}
}

// --- tools ---

func TestEvalExpr(t *testing.T) {
	tests := []struct {
		in   string
		want float64
	}{
		{"1+2*3", 7},
		{"(1+2)*3", 9},
		{"2^3^2", 512}, // right-associative: 2^(3^2)
		{"-2^2", -4},   // the power binds tighter than the minus
		{"2^-1", 0.5},
		{"10/4", 2.5},
		{"7%3", 1},
		{"  4 * ( 2 + 3 ) ", 20},
		{"sqrt(16)+abs(-2)", 6},
		{"round(2.5)+floor(1.9)+ceil(0.1)", 5},
		{"2*pi", 2 * math.Pi},
		{"e", math.E},
	}
	for _, tt := range tests {
		got, err := evalExpr(tt.in)
		if err != nil {
			t.Errorf("evalExpr(%q) failed: %v", tt.in, err)
			continue
		}
		if math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("evalExpr(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

func TestEvalExprErrors(t *testing.T) {
	for _, in := range []string{"", "2+", "(1", "1/0", "7%0", "abc", "sqrt(-1)", "2 3", "1..2", "foo(1)"} {
		if _, err := evalExpr(in); err == nil {
			t.Errorf("evalExpr(%q) should fail", in)
		}
	}
}

func TestRunTool(t *testing.T) {
	if got := runTool("calculate", map[string]interface{}{"expression": "6*7"}); got != "42" {
		t.Errorf("got %q, want 42", got)
	}
	if got := runTool("calculate", map[string]interface{}{"expression": "0.1+0.2"}); got != "0.3" {
		t.Errorf("floating point noise should be rounded away, got %q", got)
	}
	if got := runTool("calculate", map[string]interface{}{"expression": "1/0"}); !strings.HasPrefix(got, "error:") {
		t.Errorf("errors should come back as text starting with error:, got %q", got)
	}
	if got := runTool("no_such_tool", nil); !strings.HasPrefix(got, "error:") {
		t.Errorf("unknown tools should give an error, got %q", got)
	}
}

func TestToolCurrentTime(t *testing.T) {
	got, err := toolCurrentTime(map[string]interface{}{"timezone": "UTC"})
	if err != nil || !strings.Contains(got, "UTC") {
		t.Errorf("got %q, %v", got, err)
	}
	if _, err := toolCurrentTime(map[string]interface{}{"timezone": "Not/AZone"}); err == nil {
		t.Error("an unknown time zone should be an error")
	}
	if got, err := toolCurrentTime(nil); err != nil || got == "" {
		t.Errorf("no arguments should give the local time, got %q, %v", got, err)
	}
}

func TestToolDefs(t *testing.T) {
	defs := toolDefs()
	if len(defs) != len(toolSpecs) {
		t.Fatalf("got %d definitions for %d tools", len(defs), len(toolSpecs))
	}
	if _, err := json.Marshal(defs); err != nil {
		t.Fatalf("definitions must be valid JSON: %v", err)
	}
	names := map[string]bool{}
	for _, d := range defs {
		if d.Type != "function" || d.Function.Description == "" {
			t.Errorf("incomplete definition: %+v", d)
		}
		names[d.Function.Name] = true
	}
	for _, want := range []string{"get_current_time", "calculate", "list_files", "read_file"} {
		if !names[want] {
			t.Errorf("tool %q is missing", want)
		}
	}
}

func TestBuildChatMessagesWithTools(t *testing.T) {
	call := ToolCall{Function: ToolCallFunction{Name: "get_current_time", Arguments: map[string]interface{}{"timezone": "UTC"}}}
	history := []ChatMessage{
		{Role: "You", Content: "what time is it?"},
		{Role: "AI", ToolCalls: []ToolCall{call}}, // no text, only a tool request: must still be sent
		{Role: "Tool", ToolName: "get_current_time", Content: "Monday, 12:00"},
	}
	want := []OllamaChatMsg{
		{Role: "user", Content: "what time is it?"},
		{Role: "assistant", ToolCalls: []ToolCall{call}},
		{Role: "tool", ToolName: "get_current_time", Content: "Monday, 12:00"},
	}
	if got := buildChatMessages(history, "", nil); !reflect.DeepEqual(got, want) {
		t.Errorf("got  %+v\nwant %+v", got, want)
	}
}

func TestIsNoteMsg(t *testing.T) {
	notes := []ChatMessage{
		{Role: "System", Content: "Loaded model"},
		{Role: "Tool", ToolName: "calculate", Content: "4"},
		{Role: "AI", ToolCalls: []ToolCall{{}}},
	}
	for _, m := range notes {
		if !isNoteMsg(m) {
			t.Errorf("%+v should be a note", m)
		}
	}
	chat := []ChatMessage{
		{Role: "You", Content: ""},
		{Role: "AI", Content: "hello", ToolCalls: []ToolCall{{}}},
		{Role: "AI"},
	}
	for _, m := range chat {
		if isNoteMsg(m) {
			t.Errorf("%+v should be a normal message", m)
		}
	}
}

func TestShortenAndDescribeCalls(t *testing.T) {
	if got := shorten("  hello \n  world  ", 100); got != "hello world" {
		t.Errorf("got %q", got)
	}
	if got := shorten("abcdef", 3); got != "abc…" {
		t.Errorf("got %q", got)
	}

	noArgs := []ToolCall{{Function: ToolCallFunction{Name: "get_current_time"}}}
	if got := describeCalls(noArgs); got != "get_current_time()" {
		t.Errorf("got %q", got)
	}
	withArgs := []ToolCall{{Function: ToolCallFunction{Name: "get_current_time", Arguments: map[string]interface{}{"timezone": "UTC"}}}}
	if got := describeCalls(withArgs); got != `get_current_time({"timezone":"UTC"})` {
		t.Errorf("got %q", got)
	}
}

func TestSafePathAndFileTools(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(old)

	write := func(path, content string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(root, "a.txt"), "hi")
	write(filepath.Join(root, ".secret"), "x")
	write(filepath.Join(root, "bin.dat"), "\x00\x01\x02")
	write(filepath.Join(outside, "o.txt"), "outside")
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}

	good := []string{"a.txt", "./a.txt", "sub", "."}
	bad := []string{filepath.Join(outside, "o.txt"), "../nothing", ".secret", "missing.txt"}
	if err := os.Symlink(outside, filepath.Join(root, "link")); err == nil {
		bad = append(bad, "link/o.txt") // a symlink must not be a way out
	}
	for _, p := range good {
		if _, err := safePath(p); err != nil {
			t.Errorf("safePath(%q) should work: %v", p, err)
		}
	}
	for _, p := range bad {
		if _, err := safePath(p); err == nil {
			t.Errorf("safePath(%q) should be refused", p)
		}
	}

	if got, err := toolReadFile(map[string]interface{}{"path": "a.txt"}); err != nil || got != "hi" {
		t.Errorf("read_file a.txt = %q, %v", got, err)
	}
	if _, err := toolReadFile(map[string]interface{}{"path": "sub"}); err == nil {
		t.Error("read_file on a folder should fail")
	}
	if _, err := toolReadFile(map[string]interface{}{"path": "bin.dat"}); err == nil {
		t.Error("read_file on a binary file should fail")
	}

	out, err := toolListFiles(map[string]interface{}{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "a.txt") || !strings.Contains(out, "sub/") || strings.Contains(out, ".secret") {
		t.Errorf("list_files should show files and folders but not hidden ones, got %q", out)
	}
}

// --- cloud backends (OpenRouter / OpenAI) ---

func collect(sse string) ([]OllamaGenerateRes, error) {
	var got []OllamaGenerateRes
	err := streamOpenAI(strings.NewReader(sse), func(r OllamaGenerateRes) bool {
		got = append(got, r)
		return true
	})
	return got, err
}

func TestStreamOpenAIText(t *testing.T) {
	sse := ": OPENROUTER PROCESSING\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\",\"reasoning\":\"hmm\"}}]}\n\n" +
		"data: [DONE]\n\n"
	got, err := collect(sse)
	if err != nil {
		t.Fatal(err)
	}
	want := []OllamaGenerateRes{{Response: "Hel"}, {Response: "lo", Thinking: "hmm"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestStreamOpenAIToolCall(t *testing.T) {
	// The arguments arrive in pieces and must be glued together.
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"calculate\",\"arguments\":\"{\\\"expression\\\":\"}}]}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"2+2\\\"}\"}}]}}]}\n\n" +
		"data: [DONE]\n\n"
	got, err := collect(sse)
	if err != nil {
		t.Fatal(err)
	}
	want := []OllamaGenerateRes{{ToolCalls: []ToolCall{{
		ID:       "call_1",
		Function: ToolCallFunction{Name: "calculate", Arguments: map[string]interface{}{"expression": "2+2"}},
	}}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestStreamOpenAIErrors(t *testing.T) {
	if _, err := collect("data: {\"error\":{\"message\":\"rate limited\",\"code\":429}}\n\n"); err == nil || !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("an error chunk should become an error, got %v", err)
	}
	if _, err := collect("data: {\"error\":\"plain text error\"}\n\n"); err == nil || !strings.Contains(err.Error(), "plain text error") {
		t.Errorf("an error given as a plain string should work too, got %v", err)
	}
}

func TestApiErrorDetail(t *testing.T) {
	if got := apiErrorDetail([]byte(`{"error":{"message":"No auth credentials found","code":401}}`)); got != "No auth credentials found" {
		t.Errorf("got %q", got)
	}
	if got := apiErrorDetail([]byte("Bad gateway")); got != "Bad gateway" {
		t.Errorf("non-JSON bodies should come back as they are, got %q", got)
	}
}

func TestToOpenAIMessages(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n" + strings.Repeat("\x00", 32)))
	call := ToolCall{Function: ToolCallFunction{Name: "calculate", Arguments: map[string]interface{}{"expression": "2+2"}}}
	got := toOpenAIMessages([]OllamaChatMsg{
		{Role: "system", Content: "be nice"},
		{Role: "user", Content: "what is this?", Images: []string{png}},
		{Role: "assistant", ToolCalls: []ToolCall{call}}, // a local model's call has no id
		{Role: "tool", ToolName: "calculate", Content: "4"},
		{Role: "assistant", Content: "It is 4."},
	})
	if len(got) != 5 {
		t.Fatalf("got %d messages, want 5", len(got))
	}

	parts, ok := got[1].Content.([]oaiPart)
	if !ok || len(parts) != 2 || parts[1].Type != "image_url" || !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("images should become image_url parts, got %+v", got[1].Content)
	}

	if got[2].Content != nil || len(got[2].ToolCalls) != 1 {
		t.Fatalf("assistant tool request: %+v", got[2])
	}
	id := got[2].ToolCalls[0].ID
	if id == "" || got[2].ToolCalls[0].Function.Arguments != `{"expression":"2+2"}` {
		t.Errorf("tool call needs an id and JSON-string arguments, got %+v", got[2].ToolCalls[0])
	}
	if got[3].Role != "tool" || got[3].ToolCallID != id {
		t.Errorf("the result must point at its call (%q), got %+v", id, got[3])
	}
	if got[4].Content != "It is 4." {
		t.Errorf("got %+v", got[4])
	}

	// Ids from the provider are kept as they are.
	withID := toOpenAIMessages([]OllamaChatMsg{
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "abc", Function: ToolCallFunction{Name: "read_file"}}}},
		{Role: "tool", ToolName: "read_file", ToolCallID: "abc", Content: "x"},
	})
	if withID[0].ToolCalls[0].ID != "abc" || withID[1].ToolCallID != "abc" {
		t.Errorf("provider ids should be kept: %+v", withID)
	}
	if withID[0].ToolCalls[0].Function.Arguments != "{}" {
		t.Errorf("missing arguments should become {}, got %q", withID[0].ToolCalls[0].Function.Arguments)
	}
}

func TestImageDataURL(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n" + strings.Repeat("\x00", 32)))
	jpg := base64.StdEncoding.EncodeToString([]byte("\xff\xd8\xff\xe0" + strings.Repeat("\x00", 32)))
	junk := base64.StdEncoding.EncodeToString([]byte("hello hello hello"))
	if got := imageDataURL(png); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("got %q", got)
	}
	if got := imageDataURL(jpg); !strings.HasPrefix(got, "data:image/jpeg;base64,") {
		t.Errorf("got %q", got)
	}
	if got := imageDataURL(junk); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Errorf("unknown data should fall back to png, got %q", got)
	}
}

func TestParseCloudModels(t *testing.T) {
	router := `{"data":[
		{"id":"openai/gpt-4o","context_length":128000,"architecture":{"input_modalities":["text","image"]},"supported_parameters":["tools","temperature"]},
		{"id":"meta-llama/llama-3.1-8b-instruct:free","context_length":131072,"architecture":{"input_modalities":["text"]},"supported_parameters":["temperature"]}
	]}`
	models, err := parseCloudModels(backendOpenRouter, []byte(router))
	if err != nil || len(models) != 2 {
		t.Fatalf("got %v, %v", models, err)
	}
	// Sorted by id.
	if models[0].ID != "meta-llama/llama-3.1-8b-instruct:free" || models[0].Vision || models[0].Tools || models[0].Context != 131072 {
		t.Errorf("unexpected first model: %+v", models[0])
	}
	if models[1].ID != "openai/gpt-4o" || !models[1].Vision || !models[1].Tools || models[1].Context != 128000 {
		t.Errorf("unexpected second model: %+v", models[1])
	}

	openai := `{"data":[{"id":"gpt-4o"},{"id":"text-embedding-3-small"},{"id":"whisper-1"},{"id":"o3-mini"},{"id":"gpt-4o-realtime-preview"},{"id":"dall-e-3"}]}`
	models, err = parseCloudModels(backendOpenAI, []byte(openai))
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, m := range models {
		ids = append(ids, m.ID)
	}
	if !reflect.DeepEqual(ids, []string{"gpt-4o", "o3-mini"}) {
		t.Errorf("only chat models should stay, got %v", ids)
	}
	if !models[0].Vision || !models[0].Tools || models[0].Thinking {
		t.Errorf("gpt-4o: %+v", models[0])
	}
	if !models[1].Thinking {
		t.Errorf("o3-mini should count as a reasoning model: %+v", models[1])
	}

	if _, err := parseCloudModels(backendOpenRouter, []byte("not json")); err == nil {
		t.Error("bad JSON should be an error")
	}
}

func TestBackendHelpers(t *testing.T) {
	if backendLabel(backendOpenRouter) != "OpenRouter" || backendLabel(backendOpenAI) != "OpenAI" || backendLabel("") != "Ollama (local)" {
		t.Error("unexpected backend labels")
	}
	if cloudBase(backendOpenRouter) != "https://openrouter.ai/api/v1" || cloudBase(backendOpenAI) != "https://api.openai.com/v1" || cloudBase(backendOllama) != "" {
		t.Error("unexpected base URLs")
	}

	m := model{backend: backendOpenRouter, config: Config{OpenRouterKey: "saved"}}
	if !m.isCloud() || m.apiKey() != "saved" {
		t.Errorf("isCloud/apiKey wrong: %v %q", m.isCloud(), m.apiKey())
	}

	t.Setenv("OPENAI_API_KEY", "from-env")
	m = model{backend: backendOpenAI}
	if got := m.apiKey(); got != "from-env" {
		t.Errorf("the environment variable should be used when nothing is saved, got %q", got)
	}
	m.config.OpenAIKey = "saved"
	if got := m.apiKey(); got != "saved" {
		t.Errorf("a saved key wins over the environment, got %q", got)
	}

	if (model{}).isCloud() || (model{backend: backendOllama}).isCloud() {
		t.Error("Ollama is not a cloud backend")
	}
}

func TestCloudContextSize(t *testing.T) {
	m := model{backend: backendOpenRouter, modelMaxCtx: 32000, config: Config{NumCtx: 8192}}
	if got := m.numCtxSetting(); got != 0 {
		t.Errorf("num_ctx is an Ollama setting, got %d", got)
	}
	if got := m.effectiveCtx(); got != 32000 {
		t.Errorf("the meter should use the model's own limit, got %d", got)
	}
	m.modelMaxCtx = 0
	if got := m.effectiveCtx(); got != 128000 {
		t.Errorf("unknown limits fall back to 128000, got %d", got)
	}
}

func TestModelFilter(t *testing.T) {
	m := model{models: []string{"llama3", "gemma3:4b", "gemma3:270m", "qwen3"}}
	if got := m.visibleModels(); len(got) != 4 {
		t.Errorf("no filter shows everything, got %v", got)
	}

	m.filter = "GEMMA" // case does not matter
	if got := m.visibleModels(); !reflect.DeepEqual(got, []int{1, 2}) {
		t.Errorf("got %v, want [1 2]", got)
	}

	m.cursor = 0 // hidden by the filter: the selection jumps to the first match
	m.moveCursor(0)
	if m.cursor != 1 {
		t.Errorf("cursor = %d, want 1", m.cursor)
	}
	m.moveCursor(1)
	m.moveCursor(1) // stops at the end
	if m.cursor != 2 {
		t.Errorf("cursor = %d, want 2", m.cursor)
	}
	m.moveCursor(-1)
	m.moveCursor(-1) // stops at the start
	if m.cursor != 1 {
		t.Errorf("cursor = %d, want 1", m.cursor)
	}

	m.filter = "nothing matches this"
	m.cursor = 3
	m.moveCursor(1) // no visible models: must not crash or move
	if m.cursor != 3 {
		t.Errorf("cursor should stay put, got %d", m.cursor)
	}
}

// --- QoL: new line key, console, usage, scrolling ---

func TestIsNewlineKey(t *testing.T) {
	for _, k := range []string{"alt+enter", "ctrl+j", "shift+enter"} {
		if !isNewlineKey(k) {
			t.Errorf("%q should add a new line", k)
		}
	}
	for _, k := range []string{"enter", "tab", "a", ""} {
		if isNewlineKey(k) {
			t.Errorf("%q should not add a new line", k)
		}
	}
}

func TestParseConsole(t *testing.T) {
	tests := []struct{ in, cmd, arg string }{
		{":help", "help", ""},
		{"help", "help", ""},
		{"  :SWITCH gemma3:4b  ", "switch", "gemma3:4b"},
		{":models   qwen", "models", "qwen"},
		{":", "", ""},
		{"", "", ""},
	}
	for _, tt := range tests {
		cmd, arg := parseConsole(tt.in)
		if cmd != tt.cmd || arg != tt.arg {
			t.Errorf("parseConsole(%q) = (%q, %q), want (%q, %q)", tt.in, cmd, arg, tt.cmd, tt.arg)
		}
	}
}

func TestResolveModel(t *testing.T) {
	models := []string{"llama3.2:1b", "gemma3:4b", "gemma3:270m", "qwen3-vl:2b"}
	tests := []struct {
		in      string
		want    int
		wantErr bool
	}{
		{"gemma3:4b", 1, false},   // exact
		{"GEMMA3:270M", 2, false}, // exact, any case
		{"qwen", 3, false},        // a part of the name, only one match
		{"llama", 0, false},
		{"gemma3", -1, true}, // two matches
		{"mistral", -1, true},
		{"", -1, true},
	}
	for _, tt := range tests {
		got, err := resolveModel(tt.in, models)
		if got != tt.want || (err != nil) != tt.wantErr {
			t.Errorf("resolveModel(%q) = %d, %v; want %d (error: %v)", tt.in, got, err, tt.want, tt.wantErr)
		}
	}
}

func TestCapLines(t *testing.T) {
	lines := []string{"a", "b", "c", "d", "e"}
	if got := capLines(lines, 3); !reflect.DeepEqual(got, []string{"c", "d", "e"}) {
		t.Errorf("got %v", got)
	}
	if got := capLines(lines, 10); !reflect.DeepEqual(got, lines) {
		t.Errorf("short lists stay as they are, got %v", got)
	}
}

func TestConsoleHelpListsTheCommands(t *testing.T) {
	help := strings.Join(consoleHelp(), "\n")
	for _, want := range []string{":status", ":tokens", ":models", ":show", ":ps", ":unload", ":switch", ":stop", ":kill", ":exit"} {
		if !strings.Contains(help, want) {
			t.Errorf("help does not mention %s", want)
		}
	}
	if strings.Contains(help, ":llama") {
		t.Error("the help should keep its secrets")
	}
}

func TestKonamiStep(t *testing.T) {
	seq := []string{"up", "up", "down", "down", "left", "right", "left", "right", "b", "a"}
	progress := 0
	for i, key := range seq {
		var swallow, open bool
		progress, swallow, open = konamiStep(progress, key)
		if open {
			t.Fatalf("opened too early at key %d", i)
		}
		// Only the two letters are kept out of the text fields.
		if swallow != (key == "b" || key == "a") {
			t.Errorf("key %q: swallow = %v", key, swallow)
		}
	}
	if progress != len(seq) {
		t.Fatalf("progress = %d, want %d", progress, len(seq))
	}
	if _, swallow, open := konamiStep(progress, "enter"); !open || !swallow {
		t.Error("Enter after the full sequence should open it")
	}

	// A wrong key starts over, and normal typing never triggers anything.
	if p, _, _ := konamiStep(5, "x"); p != 0 {
		t.Errorf("a wrong key should reset, got %d", p)
	}
	if p, _, _ := konamiStep(5, "up"); p != 1 {
		t.Errorf("up can start the sequence again, got %d", p)
	}
	if _, _, open := konamiStep(0, "enter"); open {
		t.Error("Enter alone must not open anything")
	}
	if p, swallow, _ := konamiStep(0, "b"); p != 0 || swallow {
		t.Error("typing b normally must stay untouched")
	}
}

func TestStreamOpenAIUsage(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n" +
		"data: [DONE]\n\n"
	got, err := collect(sse)
	if err != nil {
		t.Fatal(err)
	}
	want := []OllamaGenerateRes{{Response: "Hi"}, {Usage: &UsageInfo{Prompt: 10, Completion: 5}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestAddUsage(t *testing.T) {
	var m model
	m.addUsage(UsageInfo{Prompt: 10, Completion: 5, TokensPerSec: 20})
	m.addUsage(UsageInfo{Prompt: 30, Completion: 7})
	if m.usagePrompt != 40 || m.usageCompletion != 12 || m.usageReplies != 2 || m.lastUsage.Prompt != 30 {
		t.Errorf("unexpected totals: %+v", m)
	}
	lines := strings.Join(m.tokenLines(), "\n")
	if !strings.Contains(lines, "40") || !strings.Contains(lines, "12") {
		t.Errorf("the token overview should show the totals, got %q", lines)
	}
	if got := strings.Join((model{}).tokenLines(), "\n"); !strings.Contains(got, "No token counts yet") {
		t.Errorf("got %q", got)
	}
}

func TestStopGeneration(t *testing.T) {
	cancelled := false
	m := model{
		isThinking:   true,
		cancelStream: func() { cancelled = true },
		streamChan:   make(chan OllamaGenerateRes),
		messages:     []ChatMessage{{Role: "You", Content: "hi"}, {Role: "AI", Content: "partial"}},
		genStart:     time.Now(),
	}
	if !m.stopGeneration() {
		t.Fatal("there was a reply to stop")
	}
	last := m.messages[len(m.messages)-1]
	if !cancelled || m.isThinking || m.streamChan != nil || !strings.HasSuffix(last.Content, "[Stopped by user]") || last.Elapsed == "" {
		t.Errorf("unexpected state after stopping: cancelled=%v %+v", cancelled, m)
	}
	if m.stopGeneration() {
		t.Error("a second stop has nothing to do")
	}
}

func TestScrollingIsNotForcedDownWhileStreaming(t *testing.T) {
	m := model{viewport: viewport.New(40, 5), focusMode: "list", autoScroll: true, selectedMsg: -1, snappedMsg: -1}
	for i := 0; i < 20; i++ {
		m.messages = append(m.messages, ChatMessage{Role: "System", Content: fmt.Sprintf("line %d", i)})
	}

	// The last message (the one being generated) is selected and followed.
	m.selectedMsg = len(m.messages) - 1
	m.updateViewport()
	if !m.viewport.AtBottom() {
		t.Fatal("should follow the end while autoscroll is on")
	}

	// The user scrolls up; more text arrives. The view must stay where it is.
	m.viewport.LineUp(4)
	m.autoScroll = false
	before := m.viewport.YOffset
	m.messages[len(m.messages)-1].Content += " more text"
	m.updateViewport()
	if m.viewport.YOffset != before {
		t.Errorf("the view was moved from %d to %d", before, m.viewport.YOffset)
	}

	// Moving the selection still brings the chosen message into view.
	m.selectedMsg = 2
	m.updateViewport()
	if m.viewport.YOffset != 2 {
		t.Errorf("YOffset = %d, want 2", m.viewport.YOffset)
	}
}

// --- console v2: system stats, :chat, saved logs, folders ---

func TestParseMeminfo(t *testing.T) {
	text := "MemTotal:        8000000 kB\nMemFree:          500000 kB\nMemAvailable:    3000000 kB\nSwapTotal:       2000000 kB\nSwapFree:        1500000 kB\n"
	total, used, swapTotal, swapUsed := parseMeminfo(text)
	if total != 8000000*1024 || used != 5000000*1024 || swapTotal != 2000000*1024 || swapUsed != 500000*1024 {
		t.Errorf("got %d %d %d %d", total, used, swapTotal, swapUsed)
	}

	// Old kernels have no MemAvailable: free + buffers + cache count as available.
	old := "MemTotal: 1000 kB\nMemFree: 100 kB\nBuffers: 50 kB\nCached: 150 kB\n"
	total, used, _, _ = parseMeminfo(old)
	if total != 1000*1024 || used != 700*1024 {
		t.Errorf("fallback: got %d %d", total, used)
	}
}

func TestParseProcStat(t *testing.T) {
	got, ok := parseProcStat("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8 9 10\n")
	if !ok || got.total != 1000 || got.idle != 850 {
		t.Errorf("got %+v, %v", got, ok)
	}
	if _, ok := parseProcStat("cpu a b c d e f\n"); ok {
		t.Error("garbage should not parse")
	}
	if _, ok := parseProcStat("nothing here"); ok {
		t.Error("no cpu line should not parse")
	}
}

func TestParseCPTime(t *testing.T) {
	got, ok := parseCPTime("100 0 50 10 840\n")
	if !ok || got.total != 1000 || got.idle != 840 {
		t.Errorf("got %+v, %v", got, ok)
	}
	if _, ok := parseCPTime("1 2 3"); ok {
		t.Error("too few numbers should not parse")
	}
}

func TestCPUPercent(t *testing.T) {
	prev := cpuTimes{total: 1000, idle: 850}
	tests := []struct {
		cur  cpuTimes
		want float64
	}{
		{cpuTimes{total: 2000, idle: 1350}, 50},
		{cpuTimes{total: 2000, idle: 1850}, 0},
		{cpuTimes{total: 2000, idle: 850}, 100},
		{cpuTimes{total: 1000, idle: 850}, -1}, // no time has passed
		{cpuTimes{total: 900, idle: 800}, -1},  // counters went backwards
	}
	for _, tt := range tests {
		if got := cpuPercent(prev, tt.cur); math.Abs(got-tt.want) > 1e-9 {
			t.Errorf("cpuPercent(%+v) = %v, want %v", tt.cur, got, tt.want)
		}
	}
}

func TestParseFreeBSDMemory(t *testing.T) {
	total, used := parseFreeBSDMemory("8589934592\n4096\n100000\n200000\n")
	if total != 8589934592 || used != 8589934592-300000*4096 {
		t.Errorf("got %d %d", total, used)
	}
	if total, used := parseFreeBSDMemory("1 2"); total != 0 || used != 0 {
		t.Errorf("too little data should give zeros, got %d %d", total, used)
	}
}

func TestParseSwapinfo(t *testing.T) {
	one := "Device          1K-blocks     Used    Avail Capacity\n/dev/ada0p3       8388608     1024  8387584     0%\n"
	if total, used := parseSwapinfo(one); total != 8388608*1024 || used != 1024*1024 {
		t.Errorf("one device: got %d %d", total, used)
	}
	two := "Device          1K-blocks     Used    Avail Capacity\n/dev/ada0p3 1000 100 900 10%\n/dev/ada1p3 1000 200 800 20%\nTotal 2000 300 1700 15%\n"
	if total, used := parseSwapinfo(two); total != 2000*1024 || used != 300*1024 {
		t.Errorf("two devices: got %d %d", total, used)
	}
	if total, used := parseSwapinfo(""); total != 0 || used != 0 {
		t.Errorf("no swap: got %d %d", total, used)
	}
}

func TestParseNvidiaSmi(t *testing.T) {
	gpus := parseNvidiaSmi("NVIDIA GeForce RTX 3060, 45, 3100, 12288\n\nNVIDIA Tesla, [N/A], 10, 20\n")
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus: %+v", len(gpus), gpus)
	}
	if gpus[0].Name != "NVIDIA GeForce RTX 3060" || gpus[0].Util != 45 || gpus[0].MemUsed != 3100<<20 || gpus[0].MemTotal != 12288<<20 {
		t.Errorf("first gpu: %+v", gpus[0])
	}
	if gpus[1].Util != -1 {
		t.Errorf("an unknown utilization should be -1, got %v", gpus[1].Util)
	}
	if got := parseNvidiaSmi("garbage"); len(got) != 0 {
		t.Errorf("got %+v", got)
	}
}

func TestTextBar(t *testing.T) {
	if got := textBar(0.5, 10); strings.Count(got, "█") != 5 || strings.Count(got, "░") != 5 {
		t.Errorf("got %q", got)
	}
	if got := textBar(0, 4); got != "░░░░" {
		t.Errorf("got %q", got)
	}
	if got := textBar(7, 4); got != "████" {
		t.Errorf("above 1 should clamp, got %q", got)
	}
}

func TestSystemStatsReport(t *testing.T) {
	s := SystemStats{
		CPU: 50, Cores: 4,
		MemTotal: 8 << 30, MemUsed: 4 << 30,
		SwapTotal: 2 << 30, SwapUsed: 1 << 30,
		GPUs: []GPUStat{{Name: "Test GPU", Util: 30, MemUsed: 1 << 30, MemTotal: 4 << 30}},
	}
	report := strings.Join(s.Lines(10), "\n")
	for _, want := range []string{"CPU", "RAM", "Swap", "GPU", "VRAM", "Test GPU", "50%", "4 cores"} {
		if !strings.Contains(report, want) {
			t.Errorf("the report should mention %q:\n%s", want, report)
		}
	}

	bare := strings.Join((SystemStats{CPU: -1}).Lines(10), "\n")
	for _, want := range []string{"CPU   not available", "not detected", "Swap  none"} {
		if !strings.Contains(bare, want) {
			t.Errorf("the bare report should mention %q:\n%s", want, bare)
		}
	}

	inline := s.Inline()
	if !strings.Contains(inline, "CPU 50%") || !strings.Contains(inline, "RAM 50%") || !strings.Contains(inline, "GPU 30%") {
		t.Errorf("unexpected inline text %q", inline)
	}
}

func TestToggleStats(t *testing.T) {
	var m model
	if cmd := m.toggleStats(); !m.showStats || m.statsGen != 1 || cmd == nil {
		t.Errorf("switching on should start a reading: %+v", m.statsGen)
	}
	m.haveStats = true
	if cmd := m.toggleStats(); m.showStats || m.statsGen != 2 || m.haveStats || cmd != nil {
		t.Error("switching off should stop everything and forget the numbers")
	}
}

func TestSanitizeFileName(t *testing.T) {
	tests := map[string]string{
		"My Test!":         "My_Test_",
		"../../etc/passwd": "_.._etc_passwd",
		"run-1_final.json": "run-1_final.json",
		"":                 "",
		"   ":              "",
	}
	for in, want := range tests {
		if got := sanitizeFileName(in); got != want {
			t.Errorf("sanitizeFileName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := sanitizeFileName(strings.Repeat("a", 100)); len(got) != 60 {
		t.Errorf("long names are cut to 60 characters, got %d", len(got))
	}
}

func TestRecentFiles(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.json", "b.json", "c.json", "notes.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := recentFiles(dir, 2); !reflect.DeepEqual(got, []string{"c.json", "b.json"}) {
		t.Errorf("got %v", got)
	}
	if got := recentFiles(filepath.Join(dir, "missing"), 5); len(got) != 0 {
		t.Errorf("a missing folder has no files, got %v", got)
	}
}

func TestDataFoldersAndMigration(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("the home folder is not taken from $HOME on Windows")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)

	if filepath.Base(historyDir()) != "chats" || filepath.Base(consoleLogDir()) != "console" || filepath.Dir(historyDir()) != dataDir() {
		t.Errorf("unexpected folders: %s, %s, %s", dataDir(), historyDir(), consoleLogDir())
	}

	// A chat saved by an older version lies straight in the data folder.
	if err := os.MkdirAll(dataDir(), 0o755); err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(dataDir(), "chat_20260101_000000.json")
	if err := os.WriteFile(old, []byte("[]"), 0o644); err != nil {
		t.Fatal(err)
	}
	migrateLegacyChats()
	if _, err := os.Stat(old); err == nil {
		t.Error("the old file should have been moved")
	}
	if _, err := os.Stat(filepath.Join(historyDir(), "chat_20260101_000000.json")); err != nil {
		t.Errorf("the chat should be in the chats folder: %v", err)
	}
}

func TestConsoleSave(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("the home folder is not taken from $HOME on Windows")
	}
	t.Setenv("HOME", t.TempDir())

	m := model{backend: backendOpenRouter}
	m.addUsage(UsageInfo{Prompt: 10, Completion: 5})
	m.consolePrint("hello console")
	m.consoleSave("My Test")

	data, err := os.ReadFile(filepath.Join(consoleLogDir(), "My_Test.json"))
	if err != nil {
		t.Fatalf("the log should be saved: %v", err)
	}
	var log consoleLogFile
	if err := json.Unmarshal(data, &log); err != nil {
		t.Fatal(err)
	}
	if log.SavedAt == "" || log.Backend != "OpenRouter" || log.Usage.PromptTokens != 10 || log.Usage.GeneratedTokens != 5 {
		t.Errorf("unexpected log header: %+v", log)
	}
	if len(log.Lines) == 0 || log.Lines[0] != "hello console" {
		t.Errorf("the lines should be saved: %v", log.Lines)
	}
	if last := m.consoleLines[len(m.consoleLines)-1]; !strings.Contains(last, "Saved") {
		t.Errorf("the console should say where it saved: %q", last)
	}

	m.consoleSave("") // without a name it picks one
	if got := recentFiles(consoleLogDir(), 5); len(got) != 2 {
		t.Errorf("expected two logs, got %v", got)
	}
}

func TestConsoleChatRefusals(t *testing.T) {
	last := func(m model) string { return m.consoleLines[len(m.consoleLines)-1] }

	m := model{}
	if cmd := m.consoleChat("   "); cmd != nil || !strings.Contains(last(m), "usage") {
		t.Errorf("an empty prompt should show the usage, got %q", last(m))
	}
	m = model{}
	if cmd := m.consoleChat("hi"); cmd != nil || !strings.Contains(last(m), "No model selected") {
		t.Errorf("without models: %q", last(m))
	}
	m = model{models: []string{"a"}, isThinking: true}
	if cmd := m.consoleChat("hi"); cmd != nil || !strings.Contains(last(m), ":stop") {
		t.Errorf("while chatting: %q", last(m))
	}
	t.Setenv("OPENAI_API_KEY", "") // so the test does not depend on your environment
	m = model{models: []string{"a"}, backend: backendOpenAI}
	if cmd := m.consoleChat("hi"); cmd != nil || !strings.Contains(last(m), "API key") {
		t.Errorf("a cloud model needs a key: %q", last(m))
	}
}

func TestFinishConsoleChat(t *testing.T) {
	m := model{
		backend:  backendOllama,
		models:   []string{"gemma3:4b"},
		ccChan:   make(chan OllamaGenerateRes),
		ccText:   "hello\nworld",
		ccPrompt: "hi",
		ccStart:  time.Now().Add(-2 * time.Second),
		ccFirst:  time.Now().Add(-time.Second),
		ccUsage:  &UsageInfo{Prompt: 12, Completion: 5, TokensPerSec: 25},
	}
	m.finishConsoleChat()
	out := strings.Join(m.consoleLines, "\n")
	for _, want := range []string{"hello", "world", "prompt tokens:     12", "generated tokens:  5", "25.0 tokens/s", "gemma3:4b", "first token after", "total time"} {
		if !strings.Contains(out, want) {
			t.Errorf("the result should contain %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "~") {
		t.Errorf("real counts must not be marked as estimates:\n%s", out)
	}
	if m.ccChan != nil || m.usagePrompt != 12 || m.usageCompletion != 5 {
		t.Errorf("the :chat state should be cleared and the usage counted: %+v", m.lastUsage)
	}

	// Without counts from the server the numbers are estimates.
	e := model{ccChan: make(chan OllamaGenerateRes), ccText: "abcdefgh", ccPrompt: "abcd", ccStart: time.Now()}
	e.finishConsoleChat()
	est := strings.Join(e.consoleLines, "\n")
	if !strings.Contains(est, "generated tokens:  ~2") || !strings.Contains(est, "prompt tokens:     ~1") {
		t.Errorf("estimates should be marked with ~:\n%s", est)
	}
}

func TestCancelConsoleChat(t *testing.T) {
	called := false
	m := model{ccChan: make(chan OllamaGenerateRes), ccCancel: func() { called = true }}
	if !m.cancelConsoleChat() || !called || m.ccChan != nil {
		t.Error("the running :chat should be cancelled")
	}
	if m.cancelConsoleChat() {
		t.Error("a second cancel has nothing to do")
	}
}

func TestConsoleHelpMentionsNewCommands(t *testing.T) {
	help := strings.Join(consoleHelp(), "\n")
	for _, want := range []string{":start", ":chat", ":top", ":save", ":logs"} {
		if !strings.Contains(help, want) {
			t.Errorf("help does not mention %s", want)
		}
	}
	if strings.Contains(help, ":llama") {
		t.Error("the help should keep its secrets")
	}
}
