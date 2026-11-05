package main

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
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
