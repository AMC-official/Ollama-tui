package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func lastLine(m model) string { return m.consoleLines[len(m.consoleLines)-1] }

// isolate points the home and config folders at a temporary folder, so that
// tests that save things never touch the real ones.
func isolate(t *testing.T) {
	t.Helper()
	if os.PathSeparator == '\\' {
		t.Skip("the home and config folders are not taken from $HOME on Windows")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(home, ".config"))
}

func TestConsoleColorsToggle(t *testing.T) {
	m := model{}
	m.consoleColors("off")
	if !m.plainTokens {
		t.Error("off should switch the colours off")
	}
	m.consoleColors("on")
	if m.plainTokens {
		t.Error("on should switch them on")
	}
	m.consoleColors("")
	if !m.plainTokens {
		t.Error("no argument toggles")
	}
	m.consoleColors("weird")
	if !m.plainTokens || !strings.Contains(lastLine(m), "usage") {
		t.Errorf("a bad argument should only show the usage, got %q", lastLine(m))
	}
}

func TestConsoleChatWithoutPromptDumpsTheChat(t *testing.T) {
	m := model{plainTokens: true, messages: []ChatMessage{{Role: "You", Content: "hello"}}}
	if cmd := m.runConsole("chat"); cmd != nil {
		t.Error("the dump needs no background work")
	}
	out := strings.Join(m.consoleLines, "\n")
	if !strings.Contains(out, "[1] you") || !strings.Contains(out, "hello") {
		t.Errorf(":chat without a prompt should print the conversation:\n%s", out)
	}

	m.runConsole("ctx")
	if !strings.Contains(strings.Join(m.consoleLines, "\n"), "Total:") {
		t.Error(":ctx should print the total")
	}
	m.runConsole("dump full")
	m.runConsole("colors off")
	if !m.plainTokens {
		t.Error(":colors off should work through the console")
	}
}

func TestConsoleHelpListsTheNewCommands(t *testing.T) {
	help := strings.Join(consoleHelp(), "\n")
	for _, want := range []string{":dump", ":ctx", ":colors", ":images", ":view", ":copy", ":export", ":set", ":system"} {
		if !strings.Contains(help, want) {
			t.Errorf("help does not mention %s", want)
		}
	}
}

func TestConsoleSet(t *testing.T) {
	isolate(t)
	m := model{}

	m.consoleSet("temp 0.7")
	if m.config.Temperature == nil || *m.config.Temperature != 0.7 {
		t.Errorf("temperature not set: %+v", m.config.Temperature)
	}
	m.consoleSet("temp 5")
	if !strings.Contains(lastLine(m), "0 to 2") || *m.config.Temperature != 0.7 {
		t.Errorf("a bad value must be refused and change nothing, got %q", lastLine(m))
	}
	m.consoleSet("top_p 0.9")
	m.consoleSet("top_k 40")
	m.consoleSet("ctx 2048")
	if m.config.TopP == nil || *m.config.TopP != 0.9 || m.config.TopK == nil || *m.config.TopK != 40 || m.config.NumCtx != 2048 {
		t.Errorf("settings not stored: %+v", m.config)
	}
	m.consoleSet("ctx 100")
	if m.config.NumCtx != 2048 || !strings.Contains(lastLine(m), "512") {
		t.Errorf("a context below 512 must be refused, got %q", lastLine(m))
	}
	m.modelMaxCtx = 4096
	m.consoleSet("ctx 8192")
	if m.config.NumCtx != 2048 || !strings.Contains(lastLine(m), "at most 4096") {
		t.Errorf("the model limit must be respected, got %q", lastLine(m))
	}
	m.consoleSet("temp default")
	if m.config.Temperature != nil {
		t.Error("default should clear the value")
	}
	m.consoleSet("bogus 1")
	if !strings.Contains(lastLine(m), "Unknown setting") {
		t.Errorf("got %q", lastLine(m))
	}
	m.consoleSet("temp")
	if !strings.Contains(lastLine(m), "usage") {
		t.Errorf("got %q", lastLine(m))
	}

	// The change is saved, and loadConfig sees it.
	if cfg := loadConfig(); cfg.NumCtx != 2048 || cfg.TopK == nil || *cfg.TopK != 40 {
		t.Errorf("the settings should be saved: %+v", cfg)
	}
}

func TestConsoleSystem(t *testing.T) {
	isolate(t)
	m := model{}
	m.consoleSystem("")
	if !strings.Contains(lastLine(m), "No system prompt") {
		t.Errorf("got %q", lastLine(m))
	}
	m.consoleSystem("Be brief.")
	if m.systemPrompt != "Be brief." || m.config.SystemPrompt != "Be brief." {
		t.Errorf("the prompt should be set in both places: %q / %q", m.systemPrompt, m.config.SystemPrompt)
	}
	m.consoleSystem("")
	if !strings.Contains(strings.Join(m.consoleLines, "\n"), "Be brief.") {
		t.Error("the prompt should be shown")
	}
	m.consoleSystem("clear")
	if m.systemPrompt != "" || m.config.SystemPrompt != "" {
		t.Error("clear should remove the prompt")
	}
}

func TestChatMarkdown(t *testing.T) {
	call := ToolCall{Function: ToolCallFunction{Name: "calculate"}}
	md := chatMarkdown([]ChatMessage{
		{Role: "You", Content: "What is 2+2?", Context: "Context from attached files:\nnotes\n"},
		{Role: "AI", Content: "Let me check.", Thinking: "hmm", ToolCalls: []ToolCall{call}, Elapsed: "1.2s"},
		{Role: "Tool", ToolName: "calculate", Content: "4"},
		{Role: "AI", Content: "It is 4."},
		{Role: "System", Content: "a note"},
	}, "gemma3:4b", "Ollama")
	for _, want := range []string{"# Ollama-TUI chat", "gemma3:4b", "## You", "What is 2+2?", "Attached text", "notes", "## AI", "Thinking", "calculate", "Reply took 1.2s", "### Tool result: calculate", "It is 4.", "> a note"} {
		if !strings.Contains(md, want) {
			t.Errorf("the export should contain %q:\n%s", want, md)
		}
	}
}

func TestConsoleExport(t *testing.T) {
	isolate(t)
	m := model{}
	m.consoleExport("")
	if !strings.Contains(lastLine(m), "empty") {
		t.Errorf("an empty chat has nothing to export: %q", lastLine(m))
	}
	m.messages = []ChatMessage{{Role: "You", Content: "hello export"}}
	m.consoleExport("My chat")
	data, err := os.ReadFile(filepath.Join(exportDir(), "My_chat.md"))
	if err != nil {
		t.Fatalf("the export should be written: %v", err)
	}
	if !strings.Contains(string(data), "hello export") {
		t.Errorf("unexpected export:\n%s", data)
	}
	if !strings.Contains(lastLine(m), "Exported") {
		t.Errorf("the console should say where it saved: %q", lastLine(m))
	}
}

func TestConsoleSaveStripsColours(t *testing.T) {
	isolate(t)
	m := model{}
	m.consolePrint("  \x1b[38;5;220mhello\x1b[0m")
	m.consoleSave("colours")
	data, err := os.ReadFile(filepath.Join(consoleLogDir(), "colours.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "38;5;220") || !strings.Contains(string(data), "hello") {
		t.Errorf("saved logs must be plain text: %s", data)
	}
}

func TestImageNotesAndLookup(t *testing.T) {
	m := model{messages: []ChatMessage{
		{Role: "System", Content: imageNotePrefix + "cat.png"},
		{Role: "System", Content: "just a note"},
	}}
	m.rememberImages([]Attachment{
		{Name: "cat.png", Path: "/tmp/cat.png", IsImage: true},
		{Name: "doc.txt", Path: "/tmp/doc.txt"},
	})
	imgs := m.chatImages()
	if len(imgs) != 1 || imgs[0].Path != "/tmp/cat.png" {
		t.Fatalf("expected the one sent picture, got %+v", imgs)
	}

	ref, ok := m.imageForMessage(0)
	if !ok || ref.Name != "cat.png" || ref.Path != "/tmp/cat.png" {
		t.Errorf("the note should lead to its picture: %+v %v", ref, ok)
	}
	if ref, ok := m.imageForMessage(1); !ok || ref.Name != "cat.png" {
		t.Errorf("another message falls back to the newest picture: %+v %v", ref, ok)
	}
	if _, ok := (model{}).imageForMessage(0); ok {
		t.Error("a chat without pictures has nothing to show")
	}

	// Attached but not sent yet.
	m.attachments = []Attachment{{Name: "new.jpg", Path: "/x/new.jpg", IsImage: true}}
	imgs = m.chatImages()
	if len(imgs) != 2 || imgs[1].Name != "new.jpg" {
		t.Errorf("pending pictures come last: %+v", imgs)
	}

	// A note from a chat loaded from disk: the picture is known by name only.
	old := model{messages: []ChatMessage{{Role: "System", Content: imageNotePrefix + "ghost.png"}}}
	if ref, ok := old.imageFromNote(old.messages[0].Content); !ok || ref.Path != "" || ref.Name != "ghost.png" {
		t.Errorf("unexpected: %+v %v", ref, ok)
	}
}

func TestRememberImagesIsCapped(t *testing.T) {
	m := model{}
	for i := 0; i < maxSeenImages+10; i++ {
		m.rememberImages([]Attachment{{Name: "p.png", Path: "/p.png", IsImage: true}})
	}
	if len(m.seenImages) != maxSeenImages {
		t.Errorf("expected %d, got %d", maxSeenImages, len(m.seenImages))
	}
}

func TestConsoleImagesAndView(t *testing.T) {
	m := model{}
	m.consoleImages()
	if !strings.Contains(lastLine(m), "No images") {
		t.Errorf("got %q", lastLine(m))
	}
	if cmd := m.consoleViewImage(""); cmd != nil || !strings.Contains(lastLine(m), "No images") {
		t.Errorf("without pictures :view has nothing to show: %q", lastLine(m))
	}
	if cmd := m.consoleViewImage("7"); cmd != nil || !strings.Contains(lastLine(m), "Pick a number") {
		t.Errorf("a wrong number should be refused: %q", lastLine(m))
	}
	if cmd := m.consoleViewImage("/does/not/exist.png"); cmd == nil {
		t.Error("a path to a picture should start the viewer (it reports the missing file itself)")
	}

	m.attachments = []Attachment{{Name: "a.png", Path: "/a.png", IsImage: true}}
	m.consoleImages()
	if out := strings.Join(m.consoleLines, "\n"); !strings.Contains(out, "1. a.png") {
		t.Errorf("the picture should be listed:\n%s", out)
	}
	if cmd := m.consoleViewImage("1"); cmd == nil {
		t.Error("a valid number should start the viewer")
	}
}

func TestOnImageMsg(t *testing.T) {
	m := model{}
	cmd := m.onImageMsg(imageDoneMsg{ref: imageRef{Name: "a.png"}, how: "viewer"})
	if cmd == nil || !strings.Contains(m.statusMsg, "default image viewer") || m.statusIsError {
		t.Errorf("unexpected status: %q (error=%v)", m.statusMsg, m.statusIsError)
	}
	m.onImageMsg(imageDoneMsg{ref: imageRef{Name: "a.png"}, how: "inline", err: os.ErrNotExist})
	if !m.statusIsError || !strings.Contains(m.statusMsg, "Could not show a.png") {
		t.Errorf("an error should be shown as one: %q", m.statusMsg)
	}

	m = model{console: true}
	if cmd := m.onImageMsg(imageDoneMsg{ref: imageRef{Name: "a.png"}, how: "inline"}); cmd != nil {
		t.Error("in the console the answer is printed, no status to clear")
	}
	if !strings.HasPrefix(lastLine(m), "✓ Showed a.png") {
		t.Errorf("got %q", lastLine(m))
	}
	if cmd := m.onImageMsg(imageReadyMsg{ref: imageRef{Name: "a.png"}, seq: "x", caption: "c"}); cmd == nil {
		t.Error("a ready picture should start the full-screen viewer")
	}
	if cmd := m.onImageMsg(imageFallbackMsg{ref: imageRef{Name: "a.png"}}); cmd == nil {
		t.Error("a fallback should open the default viewer")
	}
}

func TestIsImagePathAndClean(t *testing.T) {
	for _, p := range []string{"a.png", "A.JPG", "x/y.jpeg", "z.webp", "g.gif"} {
		if !isImagePath(p) {
			t.Errorf("%s should count as a picture", p)
		}
	}
	for _, p := range []string{"a.txt", "2", "noext"} {
		if isImagePath(p) {
			t.Errorf("%s should not count as a picture", p)
		}
	}
	if got := cleanImagePath(`  "/tmp/my\ pic.png" `); got != "/tmp/my pic.png" {
		t.Errorf("got %q", got)
	}
}
