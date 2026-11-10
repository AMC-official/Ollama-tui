package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// --- Conversation dump: what is sent to the model and what it costs ---
//
// :chat (without a prompt) and :dump print the conversation exactly as the
// model receives it, one block per message with its size in tokens and the
// text coloured token by token. :ctx prints only the sizes. All of it is
// built from buildChatMessages, so it shows the real request, including the
// text of attached files and the results of tool calls.

const dumpBodyLines = 40 // lines of one message shown unless :dump full is used

type dumpOptions struct {
	bodies bool // print the text of every message, not only its size
	full   bool // no cap on the lines per message
}

// dumpEntry is one part of a request: a message, or the tool definitions.
type dumpEntry struct {
	label      string
	tokens     []string
	bodyHidden bool // only the size is shown (used for the tool definitions)
}

// entryLabel is the short name shown in front of a message.
func entryLabel(c OllamaChatMsg) string {
	switch c.Role {
	case "user":
		return "you"
	case "assistant":
		return "ai"
	case "tool":
		if c.ToolName != "" {
			return "tool:" + c.ToolName
		}
	}
	return c.Role
}

// entryText is everything a message adds to the request: its text plus the
// tool calls it carries.
func entryText(c OllamaChatMsg) string {
	text := c.Content
	if len(c.ToolCalls) > 0 {
		if data, err := json.Marshal(c.ToolCalls); err == nil {
			if text != "" {
				text += "\n"
			}
			text += "tool_calls: " + string(data)
		}
	}
	return text
}

// requestEntries turns the messages of a request into dump entries.
func requestEntries(msgs []OllamaChatMsg) []dumpEntry {
	entries := make([]dumpEntry, 0, len(msgs))
	for _, c := range msgs {
		entries = append(entries, dumpEntry{label: entryLabel(c), tokens: splitTokens(entryText(c))})
	}
	return entries
}

// toolDefEntry is the cost of the tool definitions, which are sent along with
// every request while Tools mode is on (an easy thing to forget about).
func (m model) toolDefEntry() (dumpEntry, bool) {
	if !m.useTools || !m.supportsTools {
		return dumpEntry{}, false
	}
	data, err := json.Marshal(toolDefs())
	if err != nil {
		return dumpEntry{}, false
	}
	return dumpEntry{label: "tools", tokens: splitTokens(string(data)), bodyHidden: true}, true
}

// bodyLines paints the tokens and splits them into lines. A limit above zero
// cuts the text and ends with a line that says how much was left out.
func bodyLines(tokens []string, plain bool, limit int) []string {
	painted, _ := paintTokens(tokens, 0, plain)
	lines := strings.Split(painted, "\n")
	if len(lines) == 1 && lines[0] == "" {
		return []string{"(empty)"}
	}
	if limit > 0 && len(lines) > limit {
		hidden := len(lines) - limit
		lines = append(lines[:limit:limit], fmt.Sprintf("… %d more lines (:dump full shows everything)", hidden))
	}
	return lines
}

// indentLines puts a prefix in front of every line.
func indentLines(lines []string, prefix string) []string {
	out := make([]string, len(lines))
	for i, l := range lines {
		out[i] = prefix + l
	}
	return out
}

// requestBlockLines shows what was sent for a one-off :chat prompt.
func requestBlockLines(msgs []OllamaChatMsg, plain bool) []string {
	lines := []string{"Request (what was sent to the model):"}
	for _, e := range requestEntries(msgs) {
		lines = append(lines, fmt.Sprintf("  %s  ≈ %d tokens", e.label, len(e.tokens)))
		lines = append(lines, indentLines(bodyLines(e.tokens, plain, dumpBodyLines), "    ")...)
	}
	return lines
}

// dumpLines builds the whole conversation view.
func (m model) dumpLines(opt dumpOptions) []string {
	entries := requestEntries(buildChatMessages(m.messages, m.systemPrompt, nil))
	if def, ok := m.toolDefEntry(); ok {
		entries = append(entries, def)
	}
	if len(entries) == 0 {
		return []string{"The chat is empty: nothing would be sent to the model yet."}
	}

	window := m.effectiveCtx()
	limit := dumpBodyLines
	if opt.full {
		limit = 0
	}

	legend := "Every yellow or green piece is one token (an estimate: the real tokenizer is the model's own)."
	if m.plainTokens {
		legend = "Token colours are off (:colors on). Sizes are estimates: the real tokenizer is the model's own."
	}
	lines := []string{
		"Conversation as the model receives it (" + m.currentModelName() + ", " + backendLabel(m.backend) + "):",
		legend,
		"",
	}

	total := 0
	for i, e := range entries {
		n := len(e.tokens)
		total += n
		pct := float64(n) / float64(window) * 100
		lines = append(lines, fmt.Sprintf("[%d] %-12s ≈ %5d tokens %5.1f%%  %s", i+1, e.label, n, pct, colorBar(pct/100, 12, colorCyan)))
		if opt.bodies && !e.bodyHidden {
			lines = append(lines, indentLines(bodyLines(e.tokens, m.plainTokens, limit), "    ")...)
		}
	}

	lines = append(lines, "")
	totalPct := float64(total) / float64(window) * 100
	barColor := colorGreen
	if totalPct > 75 {
		barColor = colorYellow
	}
	if totalPct > 90 {
		barColor = colorRed
	}
	lines = append(lines,
		fmt.Sprintf("Total: ≈ %d of %d tokens (%.0f%%)  %s", total, window, totalPct, colorBar(totalPct/100, 20, barColor)))

	if len(entries) > 1 && total > 0 {
		order := make([]int, len(entries))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool {
			return len(entries[order[a]].tokens) > len(entries[order[b]].tokens)
		})
		var top []string
		for _, i := range order {
			if len(top) == 3 {
				break
			}
			share := float64(len(entries[i].tokens)) / float64(total) * 100
			top = append(top, fmt.Sprintf("[%d] %s %.0f%%", i+1, entries[i].label, share))
		}
		lines = append(lines, "Biggest: "+strings.Join(top, ", "))
	}

	if m.usageReplies > 0 && m.lastUsage.Prompt > 0 {
		lines = append(lines, fmt.Sprintf("The server counted %d prompt tokens for the last reply (it may be older than this chat).", m.lastUsage.Prompt))
	}
	if n := len(m.seenImages); n > 0 {
		lines = append(lines, fmt.Sprintf("%d image(s) were attached in this chat: they cost the model extra tokens that are not counted here.", n))
	}
	return lines
}

// trimBlankLines drops the empty lines at the start and the end (colour codes
// do not count as text).
func trimBlankLines(lines []string) []string {
	blank := func(l string) bool { return strings.TrimSpace(stripANSI(l)) == "" }
	for len(lines) > 0 && blank(lines[0]) {
		lines = lines[1:]
	}
	for len(lines) > 0 && blank(lines[len(lines)-1]) {
		lines = lines[:len(lines)-1]
	}
	return lines
}
