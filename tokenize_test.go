package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplitTokens(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"a  b", []string{"a", " ", " b"}},
		{"line1\nline2", []string{"line", "1", "\n", "line", "2"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		{"12345", []string{"123", "45"}},
		{"internationalization", []string{"inter", "natio", "naliz", "ation"}},
		{"...", []string{"..."}},
		{"....", []string{"...", "."}},
		{" hi", []string{" hi"}},
		{"日本語", []string{"日", "本", "語"}},
	}
	for _, tc := range tests {
		if got := splitTokens(tc.in); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("splitTokens(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if got := splitTokens(""); len(got) != 0 {
		t.Errorf("no text, no tokens: got %q", got)
	}
}

// Joining the pieces must always give the text back, whatever it contains.
func TestSplitTokensRoundTrip(t *testing.T) {
	for _, in := range []string{
		"",
		"Hello, world!",
		"  leading and trailing  ",
		"tab\tseparated\tvalues",
		"emoji 😀 test",
		"mixed 日本語 text 123456",
		"a\r\nb",
		"func main() {\n\tfmt.Println(\"hi\")\n}\n",
		"supercalifragilisticexpialidocious and more",
	} {
		if got := strings.Join(splitTokens(in), ""); got != in {
			t.Errorf("round trip of %q gave %q", in, got)
		}
	}
}

func TestApproxTokenCount(t *testing.T) {
	if n := approxTokenCount("Hello, world!"); n != 4 {
		t.Errorf("expected 4 tokens, got %d", n)
	}
	if n := approxTokenCount(""); n != 0 {
		t.Errorf("expected 0 tokens, got %d", n)
	}
}

func TestPaintTokensPlain(t *testing.T) {
	painted, n := paintTokens([]string{"a", " b", "\n", "c"}, 0, true)
	if painted != "a b\nc" || n != 4 {
		t.Errorf("plain painting should only join the tokens, got %q and %d", painted, n)
	}
	if _, n := paintTokens([]string{"x"}, 3, true); n != 4 {
		t.Errorf("the count should continue from the start, got %d", n)
	}
}

// Whatever the terminal supports, taking the colour codes away must give the
// original text back.
func TestPaintTokensKeepsText(t *testing.T) {
	text := "one two\nthree"
	painted, n := paintTokens(splitTokens(text), 0, false)
	if got := stripANSI(painted); got != text {
		t.Errorf("painting changed the text: %q", got)
	}
	if n != 4 {
		t.Errorf("expected 4 tokens (one, two, newline, three), got %d", n)
	}
	if got := stripANSI(paintText(text, false)); got != text {
		t.Errorf("paintText changed the text: %q", got)
	}
}

func TestStripANSI(t *testing.T) {
	if got := stripANSI("\x1b[38;5;220mhi\x1b[0m there"); got != "hi there" {
		t.Errorf("got %q", got)
	}
	if got := stripANSI("plain"); got != "plain" {
		t.Errorf("got %q", got)
	}
	lines := stripANSILines([]string{"\x1b[1ma\x1b[0m", "b"})
	if !reflect.DeepEqual(lines, []string{"a", "b"}) {
		t.Errorf("got %q", lines)
	}
}
