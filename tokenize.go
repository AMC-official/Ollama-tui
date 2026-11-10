package main

import (
	"strings"
	"unicode"

	"github.com/charmbracelet/lipgloss"
)

// --- Tokens: a rough splitter and the yellow/green colouring ---
//
// Ollama-TUI has no real tokenizer (every model has its own). So there are two
// kinds of tokens in the console:
//
//   - The reply of a model: one streamed chunk is (almost always) one token, so
//     those boundaries are real. They are kept in model.ccTokens.
//   - Text we only sent (system prompt, your messages, attached files): we cut
//     it with splitTokens, a rough imitation of how BPE tokenizers behave. The
//     count is close to the real one for English text, but it is an estimate.

const (
	maxWordPiece = 7 // words up to this many letters stay one token
	wordChunk    = 5 // longer words are cut into pieces of this many letters
	digitChunk   = 3 // digits are cut into groups of this many
)

// runeKind sorts one character into the classes the splitter cares about.
type runeKind int

const (
	kindLetter runeKind = iota
	kindWide            // CJK and friends: roughly one token per character
	kindDigit
	kindSpace
	kindNewline
	kindOther // punctuation, symbols, emoji
)

func kindOf(r rune) runeKind {
	switch {
	case r == '\n' || r == '\r':
		return kindNewline
	case r == ' ' || r == '\t':
		return kindSpace
	case unicode.IsDigit(r):
		return kindDigit
	case r >= 0x2E80 && unicode.IsLetter(r):
		return kindWide
	case unicode.IsLetter(r) || unicode.IsMark(r):
		return kindLetter
	}
	return kindOther
}

// splitTokens cuts text into pieces that look like what a tokenizer produces:
// a word keeps the space in front of it (" world"), digits come in small
// groups, long words are cut up, and punctuation stands on its own. Joining
// the pieces gives the original text back.
func splitTokens(s string) []string {
	rs := []rune(s)
	var out []string
	i := 0
	for i < len(rs) {
		start := i
		switch kindOf(rs[i]) {
		case kindNewline:
			for i < len(rs) && kindOf(rs[i]) == kindNewline {
				i++
			}
			out = append(out, string(rs[start:i]))

		case kindSpace:
			for i < len(rs) && kindOf(rs[i]) == kindSpace {
				i++
			}
			// A single space in front of a word belongs to the word.
			if i < len(rs) && (kindOf(rs[i]) == kindLetter || kindOf(rs[i]) == kindWide || kindOf(rs[i]) == kindDigit) {
				if i-start > 1 {
					out = append(out, string(rs[start:i-1]))
					start = i - 1
				}
				out, i = appendWord(out, rs, start, i)
			} else {
				out = append(out, string(rs[start:i]))
			}

		case kindLetter, kindWide, kindDigit:
			out, i = appendWord(out, rs, start, i)

		default: // punctuation: "..." or "---" count as one token, the rest one by one
			i++
			for i < len(rs) && i-start < 3 && rs[i] == rs[start] && kindOf(rs[i]) == kindOther {
				i++
			}
			out = append(out, string(rs[start:i]))
		}
	}
	return out
}

// appendWord adds the word that starts at rs[wordStart] (rs[start:wordStart] is
// an optional space already in front of it) and returns the position after it.
// wordStart must point at a letter, a wide character or a digit.
func appendWord(out []string, rs []rune, start, wordStart int) ([]string, int) {
	i := wordStart
	switch kindOf(rs[i]) {
	case kindWide:
		i++
		return append(out, string(rs[start:i])), i

	case kindDigit:
		for i < len(rs) && kindOf(rs[i]) == kindDigit && i-wordStart < digitChunk {
			i++
		}
		return append(out, string(rs[start:i])), i
	}

	for i < len(rs) && kindOf(rs[i]) == kindLetter {
		i++
	}
	if i-wordStart <= maxWordPiece {
		return append(out, string(rs[start:i])), i
	}
	// A long word: the first piece keeps the leading space, the rest follow.
	end := wordStart + wordChunk
	out = append(out, string(rs[start:end]))
	for end < i {
		next := end + wordChunk
		if next > i {
			next = i
		}
		out = append(out, string(rs[end:next]))
		end = next
	}
	return out, i
}

// approxTokenCount is the number of pieces splitTokens makes of s.
func approxTokenCount(s string) int {
	return len(splitTokens(s))
}

// tokenPalette returns the escape codes that switch on and off the two token
// colours (yellow, then green). It asks lipgloss instead of hard-coding the
// codes, so a terminal without colours (or NO_COLOR) gets empty strings and
// the text simply stays plain.
func tokenPalette() (pre, post [2]string) {
	colors := [2]lipgloss.Color{colorYellow, colorGreen}
	for i, c := range colors {
		// "Q" cannot be part of an escape code, so it marks where the text goes.
		marked := lipgloss.NewStyle().Foreground(c).Render("Q")
		if parts := strings.SplitN(marked, "Q", 2); len(parts) == 2 {
			pre[i], post[i] = parts[0], parts[1]
		}
	}
	return pre, post
}

// paintTokens colours the tokens alternately: yellow, green, yellow, green...
// start says how many tokens came before (so a text that is painted in several
// pieces keeps alternating). Newlines are kept out of the colour codes. It
// returns the painted text and the new token count.
func paintTokens(tokens []string, start int, plain bool) (string, int) {
	if plain {
		return strings.Join(tokens, ""), start + len(tokens)
	}
	pre, post := tokenPalette()
	var b strings.Builder
	n := start
	for _, t := range tokens {
		if t == "" {
			continue
		}
		side := n % 2
		n++
		pieces := strings.Split(t, "\n")
		for j, p := range pieces {
			if j > 0 {
				b.WriteString("\n")
			}
			if p == "" {
				continue
			}
			b.WriteString(pre[side])
			b.WriteString(p)
			b.WriteString(post[side])
		}
	}
	return b.String(), n
}

// paintText splits text with splitTokens and paints it.
func paintText(s string, plain bool) string {
	painted, _ := paintTokens(splitTokens(s), 0, plain)
	return painted
}

// stripANSI removes colour codes, so saved logs and exports stay plain text.
func stripANSI(s string) string {
	if !strings.Contains(s, "\x1b") {
		return s
	}
	var b strings.Builder
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		if rs[i] != '\x1b' {
			b.WriteRune(rs[i])
			continue
		}
		// Skip "ESC [ ... letter" (and a lone ESC).
		if i+1 < len(rs) && rs[i+1] == '[' {
			i += 2
			for i < len(rs) && !(rs[i] >= '@' && rs[i] <= '~') {
				i++
			}
		}
	}
	return b.String()
}

// stripANSILines applies stripANSI to every line.
func stripANSILines(lines []string) []string {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = stripANSI(l)
	}
	return out
}
