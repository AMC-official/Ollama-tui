package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// --- Header ---

// tokenMeter estimates how much of the context window the conversation uses
// (about 4 characters per token) and picks a colour: green, yellow or red.
func (m model) tokenMeter() (used int, percent float64, color lipgloss.Color) {
	chars := 0
	for _, msg := range m.messages {
		chars += len(msg.Content) + len(msg.Thinking) + len(msg.Context)
	}
	for _, a := range m.attachments {
		chars += len(a.Text)
	}
	used = chars / 4
	percent = float64(used) / float64(m.effectiveCtx()) * 100
	color = colorGreen
	if percent > 75 {
		color = colorYellow
	}
	if percent > 90 {
		color = colorRed
	}
	return used, percent, color
}

// compactHeader is true on small windows (phones, tiny terminals), where the
// header shrinks to a single line.
func (m model) compactHeader() bool {
	return m.width < 70 || m.height < 24
}

// headerLines is the height of the header, including its bottom border.
func (m model) headerLines() int {
	if m.compactHeader() {
		return 2
	}
	return 3
}

func (m model) headerView(modelName string) string {
	used, percent, color := m.tokenMeter()
	gray := lipgloss.NewStyle().Foreground(colorGray)
	pct := lipgloss.NewStyle().Foreground(color).Render(fmt.Sprintf(" %.0f%%", percent))

	inner := m.width - 4
	if inner < 20 {
		inner = 20
	}

	// Capability badges: what the model can do, and whether its toggle is on.
	var badges, icons []string
	if m.supportsVision {
		badges = append(badges, lipgloss.NewStyle().Foreground(colorCyan).Render("👁️ Vision"))
		icons = append(icons, "👁️")
	}
	if m.supportsThinking {
		style, word := gray, "OFF"
		if m.useThinking {
			style, word = lipgloss.NewStyle().Foreground(colorGreen), "ON"
		}
		badges = append(badges, style.Render("🧠 Think "+word))
		icons = append(icons, style.Render("🧠"))
	}
	if m.supportsTools {
		style, word := gray, "OFF"
		if m.useTools {
			style, word = lipgloss.NewStyle().Foreground(colorGreen), "ON"
		}
		badges = append(badges, style.Render("🛠️ Tools "+word))
		icons = append(icons, style.Render("🛠️"))
	}

	// Small windows: everything on one line.
	if m.compactHeader() {
		text := aiLabelStyle.Render(modelName) + " " + colorBar(percent/100, 8, color) + pct
		if len(icons) > 0 {
			text += " " + strings.Join(icons, " ")
		}
		return headerStyle.Width(m.width - 2).Render(lipgloss.NewStyle().MaxWidth(inner).Render(text))
	}

	// Line 1: model and host, badges on the right.
	host := m.endpointLabel()
	left := aiLabelStyle.Render(modelName) + gray.Render(" · "+host)
	right := strings.Join(badges, "  ")

	// Line 2: the context bar.
	barW := inner / 4
	if barW < 10 {
		barW = 10
	}
	if barW > 30 {
		barW = 30
	}
	meter := gray.Render("Context ") + colorBar(percent/100, barW, color) + pct +
		gray.Render(fmt.Sprintf("  ~%d / %d tokens", used, m.effectiveCtx()))

	line1, line2 := left, meter
	if right != "" {
		if gap := inner - lipgloss.Width(left) - lipgloss.Width(right); gap >= 2 {
			line1 = left + strings.Repeat(" ", gap) + right
		} else if gap := inner - lipgloss.Width(meter) - lipgloss.Width(right); gap >= 2 {
			line2 = meter + strings.Repeat(" ", gap) + right
		} // otherwise there is no room: the keys still work without the badges
	}

	if m.showStats && m.haveStats {
		if st := m.stats.Inline(); st != "" {
			if gap := inner - lipgloss.Width(line2) - lipgloss.Width(st); gap >= 2 {
				line2 += strings.Repeat(" ", gap) + st
			}
		}
	}
	text := lipgloss.NewStyle().MaxWidth(inner).Render(line1 + "\n" + line2)
	return headerStyle.Width(m.width - 2).Render(text)
}
