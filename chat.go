package main

import (
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// --- Small helpers for the chat box and the render tick ---

// isNewlineKey says which keys add a line break in the message box. Plain Enter
// sends; most terminals cannot tell Shift+Enter from Enter, so Alt+Enter and
// Ctrl+J work everywhere.
func isNewlineKey(s string) bool {
	return s == "alt+enter" || s == "ctrl+j" || s == "shift+enter"
}

type renderTickMsg struct{ ch chan OllamaGenerateRes }

// renderTickCmd fires ten times a second while a reply is streaming.
func renderTickCmd(ch chan OllamaGenerateRes) tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg {
		return renderTickMsg{ch: ch}
	})
}

// addUsage keeps the token counts the server reported for a reply.
func (m *model) addUsage(u UsageInfo) {
	m.usagePrompt += u.Prompt
	m.usageCompletion += u.Completion
	m.usageReplies++
	m.lastUsage = u
}

// stopGeneration cancels the reply that is streaming. It reports whether there was one.
func (m *model) stopGeneration() bool {
	if !m.isThinking {
		return false
	}
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
	return true
}

func (m model) currentModelName() string {
	if m.cursor >= 0 && m.cursor < len(m.models) {
		return m.models[m.cursor]
	}
	return "(none)"
}
