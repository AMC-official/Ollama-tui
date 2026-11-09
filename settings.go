package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// --- Settings (system prompt + sampling options) ---

// numCtxSetting is the user's context-size setting, capped to what the current
// model supports. 0 means "not set": Ollama uses its own default.
func (m model) numCtxSetting() int {
	if m.isCloud() {
		return 0 // cloud models have their own limits; num_ctx is an Ollama setting
	}
	n := m.config.NumCtx
	if n <= 0 {
		return 0
	}
	if m.modelMaxCtx > 0 && n > m.modelMaxCtx {
		n = m.modelMaxCtx
	}
	return n
}

// effectiveCtx is what the context meter divides by.
func (m model) effectiveCtx() int {
	if m.isCloud() {
		if m.modelMaxCtx > 0 {
			return m.modelMaxCtx
		}
		return 128000
	}
	if n := m.numCtxSetting(); n > 0 {
		return n
	}
	return contextWindowTokens
}

// buildOptions turns the saved settings into Ollama's "options" object.
// Anything left blank is not sent, so the model's own default applies.
func buildOptions(cfg Config, numCtx int) map[string]interface{} {
	opts := map[string]interface{}{}
	if cfg.Temperature != nil {
		opts["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		opts["top_p"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		opts["top_k"] = *cfg.TopK
	}
	if numCtx > 0 {
		opts["num_ctx"] = numCtx
	}
	if len(opts) == 0 {
		return nil
	}
	return opts
}

func formatFloat(f *float64) string {
	if f == nil {
		return ""
	}
	return strconv.FormatFloat(*f, 'f', -1, 64)
}

// openSettings fills the form from the saved config and shows it.
func (m *model) openSettings() {
	m.promptInput.SetValue(m.systemPrompt)
	vals := [4]string{formatFloat(m.config.Temperature), formatFloat(m.config.TopP), "", ""}
	if m.config.TopK != nil {
		vals[2] = strconv.Itoa(*m.config.TopK)
	}
	if m.config.NumCtx > 0 {
		vals[3] = strconv.Itoa(m.config.NumCtx)
	}
	for i := range m.settingInputs {
		m.settingInputs[i].SetValue(vals[i])
	}
	m.settingFocus = 0
	m.focusSetting()
	m.state = stateSettings
}

// focusSetting gives the keyboard to the field m.settingFocus points at.
func (m *model) focusSetting() {
	m.promptInput.Blur()
	for i := range m.settingInputs {
		m.settingInputs[i].Blur()
	}
	if m.settingFocus == 0 {
		m.promptInput.Focus()
	} else {
		m.settingInputs[m.settingFocus-1].Focus()
	}
}

func (m *model) closeSettings() {
	m.promptInput.Blur()
	for i := range m.settingInputs {
		m.settingInputs[i].Blur()
	}
	m.state = stateChat
	m.focusMode = "list"
}

// saveSettings validates the form. On a problem it stays on the screen and
// says why; otherwise it stores and saves everything.
func (m *model) saveSettings() tea.Cmd {
	cfg := m.config
	cfg.SystemPrompt = strings.TrimSpace(m.promptInput.Value())

	fail := func(msg string) tea.Cmd {
		m.statusMsg = "Not saved: " + msg
		m.statusIsError = true
		return clearStatusCmd(6 * time.Second)
	}

	if s := strings.TrimSpace(m.settingInputs[0].Value()); s == "" {
		cfg.Temperature = nil
	} else {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || v < 0 || v > 2 {
			return fail("temperature must be a number from 0 to 2")
		}
		cfg.Temperature = &v
	}

	if s := strings.TrimSpace(m.settingInputs[1].Value()); s == "" {
		cfg.TopP = nil
	} else {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil || v < 0 || v > 1 {
			return fail("top P must be a number from 0 to 1")
		}
		cfg.TopP = &v
	}

	if s := strings.TrimSpace(m.settingInputs[2].Value()); s == "" {
		cfg.TopK = nil
	} else {
		v, err := strconv.Atoi(s)
		if err != nil || v < 1 {
			return fail("top K must be a whole number, 1 or more")
		}
		cfg.TopK = &v
	}

	if s := strings.TrimSpace(m.settingInputs[3].Value()); s == "" {
		cfg.NumCtx = 0
	} else {
		v, err := strconv.Atoi(s)
		if err != nil || v < 512 {
			return fail("context tokens must be a whole number, 512 or more")
		}
		if m.modelMaxCtx > 0 && v > m.modelMaxCtx {
			return fail(fmt.Sprintf("this model supports at most %d context tokens", m.modelMaxCtx))
		}
		cfg.NumCtx = v
	}

	m.config = cfg
	m.systemPrompt = cfg.SystemPrompt
	saveConfig(cfg)
	m.closeSettings()
	m.statusMsg = "Settings saved."
	m.statusIsError = false
	return clearStatusCmd(5 * time.Second)
}
