package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// tickCmd fires once a second while a reply is being generated, so the
// elapsed-time counter in the status bar keeps updating.
func tickCmd(ch chan OllamaGenerateRes) tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg {
		return tickMsg{ch: ch}
	})
}

// generatingStatus is the status-bar text while a reply is in progress. Until
// the first token arrives it says we are waiting for the model (loading a big
// model can take a while, or even swap if it doesn't fit in RAM); afterwards it
// just shows how long the reply has been running.
func (m model) generatingStatus() string {
	secs := int(time.Since(m.genStart).Seconds())
	if m.gotOutput {
		return fmt.Sprintf("Generating... %ds (Alt+S to stop)", secs)
	}
	switch {
	case secs >= 60:
		return fmt.Sprintf("Waiting for the model... %ds • it may not fit in RAM (Alt+S to stop)", secs)
	case secs >= 20:
		return fmt.Sprintf("Waiting for the model... %ds • big models load slowly (Alt+S to stop)", secs)
	}
	return fmt.Sprintf("Waiting for the model... %ds (Alt+S to stop)", secs)
}

// --- Elapsed time / formatting helpers ---

// formatDuration prints a duration the way people read it: 3.4s, 42s, 2m 5s.
func formatDuration(d time.Duration) string {
	switch {
	case d < 10*time.Second:
		return fmt.Sprintf("%.1fs", d.Seconds())
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm %ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh %dm", int(d.Hours()), int(d.Minutes())%60)
}

func humanBytes(n int64) string {
	const unit = 1000
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "kMGTPE"[exp])
}

// colorBar draws a bar of the given width: filled cells in color, the rest grey.
func colorBar(frac float64, width int, color lipgloss.Color) string {
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	filled := int(frac * float64(width))
	return lipgloss.NewStyle().Foreground(color).Render(strings.Repeat("█", filled)) +
		lipgloss.NewStyle().Foreground(colorGray).Render(strings.Repeat("░", width-filled))
}

func progressBar(frac float64, width int) string {
	return colorBar(frac, width, colorGreen)
}
