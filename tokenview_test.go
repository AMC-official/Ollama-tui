package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestDumpLines(t *testing.T) {
	m := model{
		systemPrompt: "Be brief.",
		plainTokens:  true,
		messages: []ChatMessage{
			{Role: "You", Content: "Hello there, how are you?"},
			{Role: "AI", Content: "I am fine, thanks!"},
			{Role: "System", Content: "a note that is never sent"},
		},
	}
	out := strings.Join(m.dumpLines(dumpOptions{bodies: true}), "\n")
	for _, want := range []string{"[1] system", "[2] you", "[3] ai", "Be brief.", "Hello there, how are you?", "I am fine, thanks!", "Total:", "Biggest:"} {
		if !strings.Contains(out, want) {
			t.Errorf("the dump should contain %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "never sent") {
		t.Errorf("notes are not sent to the model, so they must not be in the dump:\n%s", out)
	}

	// :ctx shows the sizes only.
	short := strings.Join(m.dumpLines(dumpOptions{}), "\n")
	if strings.Contains(short, "Hello there") || !strings.Contains(short, "[2] you") {
		t.Errorf("the size view should not print the text:\n%s", short)
	}
}

func TestDumpLinesEmpty(t *testing.T) {
	out := strings.Join(model{}.dumpLines(dumpOptions{bodies: true}), "\n")
	if !strings.Contains(out, "empty") {
		t.Errorf("an empty chat should say so: %q", out)
	}
}

func TestDumpLinesCapsLongMessages(t *testing.T) {
	m := model{plainTokens: true, messages: []ChatMessage{{Role: "You", Content: strings.Repeat("line\n", 100)}}}
	if out := strings.Join(m.dumpLines(dumpOptions{bodies: true}), "\n"); !strings.Contains(out, "more lines") {
		t.Errorf("a long message should be cut:\n%s", out)
	}
	if out := strings.Join(m.dumpLines(dumpOptions{bodies: true, full: true}), "\n"); strings.Contains(out, "more lines") {
		t.Errorf("full must not cut anything:\n%s", out)
	}
}

func TestDumpLinesToolDefsAndServerCount(t *testing.T) {
	m := model{
		plainTokens:   true,
		useTools:      true,
		supportsTools: true,
		messages:      []ChatMessage{{Role: "You", Content: "hi"}},
	}
	m.addUsage(UsageInfo{Prompt: 42, Completion: 7})
	out := strings.Join(m.dumpLines(dumpOptions{bodies: true}), "\n")
	if !strings.Contains(out, "[2] tools") {
		t.Errorf("the tool definitions cost tokens and should be listed:\n%s", out)
	}
	if !strings.Contains(out, "42 prompt tokens") {
		t.Errorf("the server's count should be shown:\n%s", out)
	}
	m.useTools = false
	if out := strings.Join(m.dumpLines(dumpOptions{}), "\n"); strings.Contains(out, "tools") {
		t.Errorf("without Tools mode there are no definitions:\n%s", out)
	}
}

func TestRequestBlockLines(t *testing.T) {
	msgs := buildChatMessages([]ChatMessage{{Role: "You", Content: "hi there"}}, "Be brief.", nil)
	out := strings.Join(requestBlockLines(msgs, true), "\n")
	for _, want := range []string{"Request", "system", "you", "Be brief.", "hi there"} {
		if !strings.Contains(out, want) {
			t.Errorf("the request block should contain %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "~") {
		t.Errorf("the block must not use ~ (that marks estimated stats):\n%s", out)
	}
}

func TestEntryTextAndLabel(t *testing.T) {
	call := ToolCall{Function: ToolCallFunction{Name: "calculate", Arguments: map[string]interface{}{"expression": "1+1"}}}
	c := OllamaChatMsg{Role: "assistant", Content: "let me check", ToolCalls: []ToolCall{call}}
	text := entryText(c)
	if !strings.Contains(text, "let me check") || !strings.Contains(text, "calculate") {
		t.Errorf("the text and the tool calls both cost tokens: %q", text)
	}
	if entryLabel(c) != "ai" || entryLabel(OllamaChatMsg{Role: "user"}) != "you" {
		t.Error("unexpected labels")
	}
	if got := entryLabel(OllamaChatMsg{Role: "tool", ToolName: "read_file"}); got != "tool:read_file" {
		t.Errorf("got %q", got)
	}
}

func TestBodyLinesAndTrim(t *testing.T) {
	if got := bodyLines(nil, true, 0); !reflect.DeepEqual(got, []string{"(empty)"}) {
		t.Errorf("got %q", got)
	}
	got := trimBlankLines([]string{"", "  ", "a", "", "b", " ", ""})
	if !reflect.DeepEqual(got, []string{"a", "", "b"}) {
		t.Errorf("got %q", got)
	}
}
